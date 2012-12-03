/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted (faltet@pytables.org)
  Creation date: 2009-05-20

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

/*********************************************************************
  The code in this file is heavily based on FastLZ, a lightning-fast
  lossless compression library.  See LICENSES/FASTLZ.txt for details.
**********************************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "blosclz.h"

#if defined(_WIN32) && !defined(__MINGW32__)
  #include <windows.h>
  #include "win32/stdint-windows.h"
#else
  #include <stdint.h>
#endif  /* _WIN32 */


/*
 * Prevent accessing more than 8-bit at once, except on x86 architectures.
 */
#if !defined(BLOSCLZ_STRICT_ALIGN)
#define BLOSCLZ_STRICT_ALIGN
#if defined(__i386__) || defined(__386) || defined (__amd64)  /* GNU C, Sun Studio */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__i486__) || defined(__i586__) || defined(__i686__)  /* GNU C */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(_M_IX86) /* Intel, MSVC */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__386)
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(_X86_) /* MinGW */
#undef BLOSCLZ_STRICT_ALIGN
#elif defined(__I86__) /* Digital Mars */
#undef BLOSCLZ_STRICT_ALIGN
#endif
#endif

/*
 * Always check for bound when decompressing.
 * Generally it is best to leave it defined.
 */
#define BLOSCLZ_SAFE

/*
 * Give hints to the compiler for branch prediction optimization.
 */
#if defined(__GNUC__) && (__GNUC__ > 2)
#define BLOSCLZ_EXPECT_CONDITIONAL(c)    (__builtin_expect((c), 1))
#define BLOSCLZ_UNEXPECT_CONDITIONAL(c)  (__builtin_expect((c), 0))
#else
#define BLOSCLZ_EXPECT_CONDITIONAL(c)    (c)
#define BLOSCLZ_UNEXPECT_CONDITIONAL(c)  (c)
#endif

/*
 * Use inlined functions for supported systems.
 */
#if defined(__GNUC__) || defined(__DMC__) || defined(__POCC__) || defined(__WATCOMC__) || defined(__SUNPRO_C)
#define BLOSCLZ_INLINE inline
#elif defined(__BORLANDC__) || defined(_MSC_VER) || defined(__LCC__)
#define BLOSCLZ_INLINE __inline
#else
#define BLOSCLZ_INLINE
#endif

#define MAX_COPY       32
#define MAX_LEN       264  /* 256 + 8 */
#define MAX_DISTANCE 8191
#define MAX_FARDISTANCE (65535+MAX_DISTANCE-1)

#ifdef BLOSCLZ_STRICT_ALIGN
  #define BLOSCLZ_READU16(p) ((p)[0] | (p)[1]<<8)
#else
  #define BLOSCLZ_READU16(p) *((const uint16_t*)(p))
#endif


BLOSCLZ_INLINE uint32_t hash_function(uint8_t* p, uint8_t hash_log)
{
  uint32_t v;

  v = BLOSCLZ_READU16(p);
  v ^= BLOSCLZ_READU16(p+1)^(v>>(16-hash_log));
  v &= (1 << hash_log) - 1;
  return v;
}


#define IP_BOUNDARY 2

int blosclz_compress(int opt_level, const void* input,
                     int length, void* output, int maxout)
{
  uint8_t* ip = (uint8_t*) input;
  uint8_t* ibase = (uint8_t*) input;
  uint8_t* ip_bound = ip + length - IP_BOUNDARY;
  uint8_t* ip_limit = ip + length - 12;
  uint8_t* op = (uint8_t*) output;

  /* Hash table depends on the opt level.  Hash_log cannot be larger than 15. */
  uint8_t hash_log_[10] = {-1, 8, 9, 9, 11, 11, 12, 13, 14, 15};
  uint8_t hash_log = hash_log_[opt_level];
  uint16_t hash_size = 1 << hash_log;
  uint16_t *htab;
  uint8_t* op_limit;

  uint32_t hslot;
  uint32_t hval;
  uint8_t copy;

  double maxlength_[10] = {-1, .1, .15, .2, .5, .7, .85, .925, .975, 1.0};
  uint32_t maxlength = (uint32_t) (length * maxlength_[opt_level]);
  if (maxlength > (uint32_t) maxout) {
    maxlength = (uint32_t) maxout;
  }
  op_limit = op + maxlength;

  /* output buffer cannot be less than 66 bytes or we can get into problems.
     As output is usually the same length than input, we take input length. */
  if (length < 66) {
    return 0;                   /* Mark this as uncompressible */
  }

  htab = malloc(hash_size*sizeof(uint16_t));

  /* sanity check */
  if(BLOSCLZ_UNEXPECT_CONDITIONAL(length < 4)) {
    if(length) {
      /* create literal copy only */
      *op++ = length-1;
      ip_bound++;
      while(ip <= ip_bound)
        *op++ = *ip++;
      free(htab);
      return length+1;
    }
    else goto out;
  }

  /* initializes hash table */
  for (hslot = 0; hslot < hash_size; hslot++)
    htab[hslot] = 0;

  /* we start with literal copy */
  copy = 2;
  *op++ = MAX_COPY-1;
  *op++ = *ip++;
  *op++ = *ip++;

  /* main loop */
  while(BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit)) {
    const uint8_t* ref;
    uint32_t distance;
    uint32_t len = 3;         /* minimum match length */
    uint8_t* anchor = ip;  /* comparison starting-point */

    /* check for a run */
    if(ip[0] == ip[-1] && BLOSCLZ_READU16(ip-1)==BLOSCLZ_READU16(ip+1)) {
      distance = 1;
      ip += 3;
      ref = anchor - 1 + 3;
      goto match;
    }

    /* find potential match */
    hval = hash_function(ip, hash_log);
    ref = ibase + htab[hval];
    /* update hash table */
    htab[hval] = (uint16_t)(anchor - ibase);

    /* calculate distance to the match */
    distance = (uint32_t)(anchor - ref);

    /* is this a match? check the first 3 bytes */
    if (distance==0 || (distance >= MAX_FARDISTANCE) ||
        *ref++ != *ip++ || *ref++!=*ip++ || *ref++!=*ip++)
      goto literal;

    /* far, needs at least 5-byte match */
    if (distance >= MAX_DISTANCE) {
      if (*ip++ != *ref++ || *ip++ != *ref++)
        goto literal;
      len += 2;
    }

    match:

    /* last matched byte */
    ip = anchor + len;

    /* distance is biased */
    distance--;

    if(!distance) {
      /* zero distance means a run */
      uint8_t x = ip[-1];
      int64_t value, value2;
      /* Broadcast the value for every byte in a 64-bit register */
      memset(&value, x, 8);
      /* safe because the outer check against ip limit */
      while (ip < (ip_bound - (sizeof(int64_t) - IP_BOUNDARY))) {
        value2 = ((int64_t *)ref)[0];
        if (value != value2) {
          /* Find the byte that starts to differ */
          while (ip < ip_bound) {
            if (*ref++ != x) break; else ip++;
          }
          break;
        }
        else {
          ip += 8;
          ref += 8;
        }
      }
      if (ip > ip_bound) {
        long l = (long)(ip - ip_bound);
        ip -= l;
        ref -= l;
      }   /* End of optimization */
    }
    else {
      for(;;) {
        /* safe because the outer check against ip limit */
        while (ip < (ip_bound - (sizeof(int64_t) - IP_BOUNDARY))) {
          if (*ref++ != *ip++) break;
          if (((int64_t *)ref)[0] != ((int64_t *)ip)[0]) {
            /* Find the byte that starts to differ */
            while (ip < ip_bound) {
              if (*ref++ != *ip++) break;
            }
            break;
          }
          else {
            ip += 8;
            ref += 8;
          }
        }
        /* Last correction before exiting loop */
        if (ip > ip_bound) {
          uint32_t l = (uint32_t)(ip - ip_bound);
          ip -= l;
          ref -= l;
        }   /* End of optimization */
        break;
      }
    }

    /* if we have copied something, adjust the copy count */
    if (copy)
      /* copy is biased, '0' means 1 byte copy */
      *(op-copy-1) = copy-1;
    else
      /* back, to overwrite the copy count */
      op--;

    /* reset literal counter */
    copy = 0;

    /* length is biased, '1' means a match of 3 bytes */
    ip -= 3;
    len = (uint32_t)(ip - anchor);

    /* check that we have space enough to encode the match for all the cases */
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+(len/255)+6 > op_limit)) goto out;

    /* encode the match */
    if(distance < MAX_DISTANCE) {
      if(len < 7) {
        *op++ = (len << 5) + (distance >> 8);
        *op++ = (distance & 255);
      }
      else {
        *op++ = (uint8_t)((7 << 5) + (distance >> 8));
        for(len-=7; len >= 255; len-= 255)
          *op++ = 255;
        *op++ = len;
        *op++ = (distance & 255);
      }
    }
    else {
      /* far away, but not yet in the another galaxy... */
      if(len < 7) {
        distance -= MAX_DISTANCE;
        *op++ = (uint8_t)((len << 5) + 31);
        *op++ = 255;
        *op++ = (uint8_t)(distance >> 8);
        *op++ = distance & 255;
      }
      else {
        distance -= MAX_DISTANCE;
        *op++ = (7 << 5) + 31;
        for(len-=7; len >= 255; len-= 255)
          *op++ = 255;
        *op++ = len;
        *op++ = 255;
        *op++ = (uint8_t)(distance >> 8);
        *op++ = distance & 255;
      }
    }

    /* update the hash at match boundary */
    hval = hash_function(ip, hash_log);
    htab[hval] = (uint16_t)(ip++ - ibase);
    hval = hash_function(ip, hash_log);
    htab[hval] = (uint16_t)(ip++ - ibase);

    /* assuming literal copy */
    *op++ = MAX_COPY-1;

    continue;

  literal:
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+2 > op_limit)) goto out;
    *op++ = *anchor++;
    ip = anchor;
    copy++;
    if(BLOSCLZ_UNEXPECT_CONDITIONAL(copy == MAX_COPY)) {
      copy = 0;
      *op++ = MAX_COPY-1;
    }
  }

  /* left-over as literal copy */
  ip_bound++;
  while(ip <= ip_bound) {
    if (BLOSCLZ_UNEXPECT_CONDITIONAL(op+2 > op_limit)) goto out;
    *op++ = *ip++;
    copy++;
    if(copy == MAX_COPY) {
      copy = 0;
      *op++ = MAX_COPY-1;
    }
  }

  /* if we have copied something, adjust the copy length */
  if(copy)
    *(op-copy-1) = copy-1;
  else
    op--;

  /* marker for blosclz */
  *(uint8_t*)output |= (1 << 5);

  free(htab);
  return (int)(op - (uint8_t*)output);

 out:
  free(htab);
  return 0;

}


int blosclz_decompress(const void* input, int length, void* output, int maxout)
{
  const uint8_t* ip = (const uint8_t*) input;
  const uint8_t* ip_limit  = ip + length;
  uint8_t* op = (uint8_t*) output;
  uint8_t* op_limit = op + maxout;
  uint32_t ctrl = (*ip++) & 31;
  uint32_t loop = 1;

  do {
    const uint8_t* ref = op;
    uint32_t len = ctrl >> 5;
    uint32_t ofs = (ctrl & 31) << 8;

    if(ctrl >= 32) {
      uint8_t code;
      len--;
      ref -= ofs;
      if (len == 7-1)
        do {
          code = *ip++;
          len += code;
        } while (code==255);
      code = *ip++;
      ref -= code;

      /* match from 16-bit distance */
      if(BLOSCLZ_UNEXPECT_CONDITIONAL(code==255))
      if(BLOSCLZ_EXPECT_CONDITIONAL(ofs==(31 << 8))) {
        ofs = (*ip++) << 8;
        ofs += *ip++;
        ref = op - ofs - MAX_DISTANCE;
      }

#ifdef BLOSCLZ_SAFE
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + len + 3 > op_limit)) {
        return 0;
      }

      if (BLOSCLZ_UNEXPECT_CONDITIONAL(ref-1 < (uint8_t *)output)) {
        return 0;
      }
#endif

      if(BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit))
        ctrl = *ip++;
      else
        loop = 0;

      if(ref == op) {
        /* optimize copy for a run */
        uint8_t b = ref[-1];
        memset(op, b, len+3);
        op += len+3;
      }
      else {
        /* copy from reference */
        ref--;
        len += 3;
        if (abs((int32_t)(ref-op)) <= (int32_t)len) {
          /* src and dst do overlap: do a loop */
          for(; len; --len)
            *op++ = *ref++;
          /* The memmove below does not work well (don't know why) */
          /* memmove(op, ref, len);
             op += len;
             ref += len;
             len = 0; */
        }
        else {
          memcpy(op, ref, len);
          op += len;
          ref += len;
        }
      }
    }
    else {
      ctrl++;
#ifdef BLOSCLZ_SAFE
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(op + ctrl > op_limit)) {
        return 0;
      }
      if (BLOSCLZ_UNEXPECT_CONDITIONAL(ip + ctrl > ip_limit)) {
        return 0;
      }
#endif

      memcpy(op, ip, ctrl);
      ip += ctrl;
      op += ctrl;

      loop = BLOSCLZ_EXPECT_CONDITIONAL(ip < ip_limit);
      if(loop)
        ctrl = *ip++;
    }
  } while(BLOSCLZ_EXPECT_CONDITIONAL(loop));

  return (int)(op - (uint8_t*)output);
}
