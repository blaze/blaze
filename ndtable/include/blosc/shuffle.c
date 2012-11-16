/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted (faltet@pytables.org)
  Creation date: 2009-05-20

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

#include <stdio.h>
#include <string.h>
#include "shuffle.h"

#if defined(_WIN32) && !defined(__MINGW32__)
  #include <windows.h>
  #include "win32/stdint-windows.h"
  #define __SSE2__          /* Windows does not define this by default */
#else
  #include <stdint.h>
  #include <inttypes.h>
#endif  /* _WIN32 */


/* The non-SSE2 versions of shuffle and unshuffle */

/* Shuffle a block.  This can never fail. */
void _shuffle(size_t bytesoftype, size_t blocksize,
	      uint8_t* _src, uint8_t* _dest) {
  size_t i, j, neblock, leftover;

  /* Non-optimized shuffle */
  neblock = blocksize / bytesoftype;  /* Number of elements in a block */
  for (j = 0; j < bytesoftype; j++) {
    for (i = 0; i < neblock; i++) {
      _dest[j*neblock+i] = _src[i*bytesoftype+j];
    }
  }
  leftover = blocksize % bytesoftype;
  memcpy(_dest + neblock*bytesoftype, _src + neblock*bytesoftype, leftover);
}

/* Unshuffle a block.  This can never fail. */
void _unshuffle(size_t bytesoftype, size_t blocksize,
		uint8_t* _src, uint8_t* _dest) {
  size_t i, j, neblock, leftover;

  /* Non-optimized unshuffle */
  neblock = blocksize / bytesoftype;  /* Number of elements in a block */
  for (i = 0; i < neblock; i++) {
    for (j = 0; j < bytesoftype; j++) {
      _dest[i*bytesoftype+j] = _src[j*neblock+i];
    }
  }
  leftover = blocksize % bytesoftype;
  memcpy(_dest+neblock*bytesoftype, _src+neblock*bytesoftype, leftover);
}


#ifdef __SSE2__

/* The SSE2 versions of shuffle and unshuffle */

#include <emmintrin.h>

/* The next is useful for debugging purposes */
void printxmm(__m128i xmm0) {
  uint8_t buf[16];

  ((__m128i *)buf)[0] = xmm0;
  printf("%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x\n",
          buf[0], buf[1], buf[2], buf[3],
          buf[4], buf[5], buf[6], buf[7],
          buf[8], buf[9], buf[10], buf[11],
          buf[12], buf[13], buf[14], buf[15]);
}


/* Routine optimized for shuffling a buffer for a type size of 2 bytes. */
static void
shuffle2(uint8_t* dest, uint8_t* src, size_t size)
{
  size_t i, j, k;
  size_t numof16belem;
  __m128i xmm0[2], xmm1[2];

  numof16belem = size / (16*2);
  for (i = 0, j = 0; i < numof16belem; i++, j += 16*2) {
    /* Fetch and transpose bytes, words and double words in groups of
       32 bytes */
    for (k = 0; k < 2; k++) {
      xmm0[k] = _mm_loadu_si128((__m128i*)(src+j+k*16));
      xmm0[k] = _mm_shufflelo_epi16(xmm0[k], 0xd8);
      xmm0[k] = _mm_shufflehi_epi16(xmm0[k], 0xd8);
      xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
      xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
      xmm0[k] = _mm_unpacklo_epi8(xmm0[k], xmm1[k]);
      xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
      xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
      xmm0[k] = _mm_unpacklo_epi16(xmm0[k], xmm1[k]);
      xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
    }
    /* Transpose quad words */
    for (k = 0; k < 1; k++) {
      xmm1[k*2] = _mm_unpacklo_epi64(xmm0[k], xmm0[k+1]);
      xmm1[k*2+1] = _mm_unpackhi_epi64(xmm0[k], xmm0[k+1]);
    }
    /* Store the result vectors */
    for (k = 0; k < 2; k++) {
      ((__m128i *)dest)[k*numof16belem+i] = xmm1[k];
    }
  }
}


/* Routine optimized for shuffling a buffer for a type size of 4 bytes. */
static void
shuffle4(uint8_t* dest, uint8_t* src, size_t size)
{
  size_t i, j, k;
  size_t numof16belem;
  __m128i xmm0[4], xmm1[4];

  numof16belem = size / (16*4);
  for (i = 0, j = 0; i < numof16belem; i++, j += 16*4) {
    /* Fetch and transpose bytes and words in groups of 64 bytes */
    for (k = 0; k < 4; k++) {
      xmm0[k] = _mm_loadu_si128((__m128i*)(src+j+k*16));
      xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0xd8);
      xmm0[k] = _mm_shuffle_epi32(xmm0[k], 0x8d);
      xmm0[k] = _mm_unpacklo_epi8(xmm1[k], xmm0[k]);
      xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x04e);
      xmm0[k] = _mm_unpacklo_epi16(xmm0[k], xmm1[k]);
    }
    /* Transpose double words */
    for (k = 0; k < 2; k++) {
      xmm1[k*2] = _mm_unpacklo_epi32(xmm0[k*2], xmm0[k*2+1]);
      xmm1[k*2+1] = _mm_unpackhi_epi32(xmm0[k*2], xmm0[k*2+1]);
    }
    /* Transpose quad words */
    for (k = 0; k < 2; k++) {
      xmm0[k*2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k+2]);
      xmm0[k*2+1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k+2]);
    }
    /* Store the result vectors */
    for (k = 0; k < 4; k++) {
      ((__m128i *)dest)[k*numof16belem+i] = xmm0[k];
    }
  }
}


/* Routine optimized for shuffling a buffer for a type size of 8 bytes. */
static void
shuffle8(uint8_t* dest, uint8_t* src, size_t size)
{
  size_t i, j, k, l;
  size_t numof16belem;
  __m128i xmm0[8], xmm1[8];

  numof16belem = size / (16*8);
  for (i = 0, j = 0; i < numof16belem; i++, j += 16*8) {
    /* Fetch and transpose bytes in groups of 128 bytes */
    for (k = 0; k < 8; k++) {
      xmm0[k] = _mm_loadu_si128((__m128i*)(src+j+k*16));
      xmm1[k] = _mm_shuffle_epi32(xmm0[k], 0x4e);
      xmm1[k] = _mm_unpacklo_epi8(xmm0[k], xmm1[k]);
    }
    /* Transpose words */
    for (k = 0, l = 0; k < 4; k++, l +=2) {
      xmm0[k*2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l+1]);
      xmm0[k*2+1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l+1]);
    }
    /* Transpose double words */
    for (k = 0, l = 0; k < 4; k++, l++) {
      if (k == 2) l += 2;
      xmm1[k*2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l+2]);
      xmm1[k*2+1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l+2]);
    }
    /* Transpose quad words */
    for (k = 0; k < 4; k++) {
      xmm0[k*2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k+4]);
      xmm0[k*2+1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k+4]);
    }
    /* Store the result vectors */
    for (k = 0; k < 8; k++) {
      ((__m128i *)dest)[k*numof16belem+i] = xmm0[k];
    }
  }
}


/* Routine optimized for shuffling a buffer for a type size of 16 bytes. */
static void
shuffle16(uint8_t* dest, uint8_t* src, size_t size)
{
  size_t i, j, k, l;
  size_t numof16belem;
  __m128i xmm0[16], xmm1[16];

  numof16belem = size / (16*16);
  for (i = 0, j = 0; i < numof16belem; i++, j += 16*16) {
    /* Fetch elements in groups of 256 bytes */
    for (k = 0; k < 16; k++) {
      xmm0[k] = _mm_loadu_si128((__m128i*)(src+j+k*16));
    }
    /* Transpose bytes */
    for (k = 0, l = 0; k < 8; k++, l +=2) {
      xmm1[k*2] = _mm_unpacklo_epi8(xmm0[l], xmm0[l+1]);
      xmm1[k*2+1] = _mm_unpackhi_epi8(xmm0[l], xmm0[l+1]);
    }
    /* Transpose words */
    for (k = 0, l = -2; k < 8; k++, l++) {
      if ((k%2) == 0) l += 2;
      xmm0[k*2] = _mm_unpacklo_epi16(xmm1[l], xmm1[l+2]);
      xmm0[k*2+1] = _mm_unpackhi_epi16(xmm1[l], xmm1[l+2]);
    }
    /* Transpose double words */
    for (k = 0, l = -4; k < 8; k++, l++) {
      if ((k%4) == 0) l += 4;
      xmm1[k*2] = _mm_unpacklo_epi32(xmm0[l], xmm0[l+4]);
      xmm1[k*2+1] = _mm_unpackhi_epi32(xmm0[l], xmm0[l+4]);
    }
    /* Transpose quad words */
    for (k = 0; k < 8; k++) {
      xmm0[k*2] = _mm_unpacklo_epi64(xmm1[k], xmm1[k+8]);
      xmm0[k*2+1] = _mm_unpackhi_epi64(xmm1[k], xmm1[k+8]);
    }
    /* Store the result vectors */
    for (k = 0; k < 16; k++) {
      ((__m128i *)dest)[k*numof16belem+i] = xmm0[k];
    }
  }
}


/* Shuffle a block.  This can never fail. */
void shuffle(size_t bytesoftype, size_t blocksize,
             uint8_t* _src, uint8_t* _dest) {
  int unaligned_dest = (int)((uintptr_t)_dest % 16);
  int power_of_two = (blocksize & (blocksize - 1)) == 0;
  int too_small = (blocksize < 256);

  if (unaligned_dest || !power_of_two || too_small) {
    /* _dest buffer is not aligned, not a power of two or is too
       small.  Call the non-sse2 version. */
    _shuffle(bytesoftype, blocksize, _src, _dest);
    return;
  }

  /* Optimized shuffle */
  /* The buffer must be aligned on a 16 bytes boundary, have a power */
  /* of 2 size and be larger or equal than 256 bytes. */
  if (bytesoftype == 4) {
    shuffle4(_dest, _src, blocksize);
  }
  else if (bytesoftype == 8) {
    shuffle8(_dest, _src, blocksize);
  }
  else if (bytesoftype == 16) {
    shuffle16(_dest, _src, blocksize);
  }
  else if (bytesoftype == 2) {
    shuffle2(_dest, _src, blocksize);
  }
  else {
    /* Non-optimized shuffle */
    _shuffle(bytesoftype, blocksize, _src, _dest);
  }
}


/* Routine optimized for unshuffling a buffer for a type size of 2 bytes. */
static void
unshuffle2(uint8_t* dest, uint8_t* orig, size_t size)
{
  size_t i, k;
  size_t neblock, numof16belem;
  __m128i xmm1[2], xmm2[2];

  neblock = size / 2;
  numof16belem = neblock / 16;
  for (i = 0, k = 0; i < numof16belem; i++, k += 2) {
    /* Load the first 32 bytes in 2 XMM registrers */
    xmm1[0] = ((__m128i *)orig)[0*numof16belem+i];
    xmm1[1] = ((__m128i *)orig)[1*numof16belem+i];
    /* Shuffle bytes */
    /* Compute the low 32 bytes */
    xmm2[0] = _mm_unpacklo_epi8(xmm1[0], xmm1[1]);
    /* Compute the hi 32 bytes */
    xmm2[1] = _mm_unpackhi_epi8(xmm1[0], xmm1[1]);
    /* Store the result vectors in proper order */
    ((__m128i *)dest)[k+0] = xmm2[0];
    ((__m128i *)dest)[k+1] = xmm2[1];
  }
}


/* Routine optimized for unshuffling a buffer for a type size of 4 bytes. */
static void
unshuffle4(uint8_t* dest, uint8_t* orig, size_t size)
{
  size_t i, j, k;
  size_t neblock, numof16belem;
  __m128i xmm0[4], xmm1[4];

  neblock = size / 4;
  numof16belem = neblock / 16;
  for (i = 0, k = 0; i < numof16belem; i++, k += 4) {
    /* Load the first 64 bytes in 4 XMM registrers */
    for (j = 0; j < 4; j++) {
      xmm0[j] = ((__m128i *)orig)[j*numof16belem+i];
    }
    /* Shuffle bytes */
    for (j = 0; j < 2; j++) {
      /* Compute the low 32 bytes */
      xmm1[j] = _mm_unpacklo_epi8(xmm0[j*2], xmm0[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm1[2+j] = _mm_unpackhi_epi8(xmm0[j*2], xmm0[j*2+1]);
    }
    /* Shuffle 2-byte words */
    for (j = 0; j < 2; j++) {
      /* Compute the low 32 bytes */
      xmm0[j] = _mm_unpacklo_epi16(xmm1[j*2], xmm1[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm0[2+j] = _mm_unpackhi_epi16(xmm1[j*2], xmm1[j*2+1]);
    }
    /* Store the result vectors in proper order */
    ((__m128i *)dest)[k+0] = xmm0[0];
    ((__m128i *)dest)[k+1] = xmm0[2];
    ((__m128i *)dest)[k+2] = xmm0[1];
    ((__m128i *)dest)[k+3] = xmm0[3];
  }
}


/* Routine optimized for unshuffling a buffer for a type size of 8 bytes. */
static void
unshuffle8(uint8_t* dest, uint8_t* orig, size_t size)
{
  size_t i, j, k;
  size_t neblock, numof16belem;
  __m128i xmm0[8], xmm1[8];

  neblock = size / 8;
  numof16belem = neblock / 16;
  for (i = 0, k = 0; i < numof16belem; i++, k += 8) {
    /* Load the first 64 bytes in 8 XMM registrers */
    for (j = 0; j < 8; j++) {
      xmm0[j] = ((__m128i *)orig)[j*numof16belem+i];
    }
    /* Shuffle bytes */
    for (j = 0; j < 4; j++) {
      /* Compute the low 32 bytes */
      xmm1[j] = _mm_unpacklo_epi8(xmm0[j*2], xmm0[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm1[4+j] = _mm_unpackhi_epi8(xmm0[j*2], xmm0[j*2+1]);
    }
    /* Shuffle 2-byte words */
    for (j = 0; j < 4; j++) {
      /* Compute the low 32 bytes */
      xmm0[j] = _mm_unpacklo_epi16(xmm1[j*2], xmm1[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm0[4+j] = _mm_unpackhi_epi16(xmm1[j*2], xmm1[j*2+1]);
    }
    /* Shuffle 4-byte dwords */
    for (j = 0; j < 4; j++) {
      /* Compute the low 32 bytes */
      xmm1[j] = _mm_unpacklo_epi32(xmm0[j*2], xmm0[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm1[4+j] = _mm_unpackhi_epi32(xmm0[j*2], xmm0[j*2+1]);
    }
    /* Store the result vectors in proper order */
    ((__m128i *)dest)[k+0] = xmm1[0];
    ((__m128i *)dest)[k+1] = xmm1[4];
    ((__m128i *)dest)[k+2] = xmm1[2];
    ((__m128i *)dest)[k+3] = xmm1[6];
    ((__m128i *)dest)[k+4] = xmm1[1];
    ((__m128i *)dest)[k+5] = xmm1[5];
    ((__m128i *)dest)[k+6] = xmm1[3];
    ((__m128i *)dest)[k+7] = xmm1[7];
  }
}


/* Routine optimized for unshuffling a buffer for a type size of 16 bytes. */
static void
unshuffle16(uint8_t* dest, uint8_t* orig, size_t size)
{
  size_t i, j, k;
  size_t neblock, numof16belem;
  __m128i xmm1[16], xmm2[16];

  neblock = size / 16;
  numof16belem = neblock / 16;
  for (i = 0, k = 0; i < numof16belem; i++, k += 16) {
    /* Load the first 128 bytes in 16 XMM registrers */
    for (j = 0; j < 16; j++) {
      xmm1[j] = ((__m128i *)orig)[j*numof16belem+i];
    }
    /* Shuffle bytes */
    for (j = 0; j < 8; j++) {
      /* Compute the low 32 bytes */
      xmm2[j] = _mm_unpacklo_epi8(xmm1[j*2], xmm1[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm2[8+j] = _mm_unpackhi_epi8(xmm1[j*2], xmm1[j*2+1]);
    }
    /* Shuffle 2-byte words */
    for (j = 0; j < 8; j++) {
      /* Compute the low 32 bytes */
      xmm1[j] = _mm_unpacklo_epi16(xmm2[j*2], xmm2[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm1[8+j] = _mm_unpackhi_epi16(xmm2[j*2], xmm2[j*2+1]);
    }
    /* Shuffle 4-byte dwords */
    for (j = 0; j < 8; j++) {
      /* Compute the low 32 bytes */
      xmm2[j] = _mm_unpacklo_epi32(xmm1[j*2], xmm1[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm2[8+j] = _mm_unpackhi_epi32(xmm1[j*2], xmm1[j*2+1]);
    }
    /* Shuffle 8-byte qwords */
    for (j = 0; j < 8; j++) {
      /* Compute the low 32 bytes */
      xmm1[j] = _mm_unpacklo_epi64(xmm2[j*2], xmm2[j*2+1]);
      /* Compute the hi 32 bytes */
      xmm1[8+j] = _mm_unpackhi_epi64(xmm2[j*2], xmm2[j*2+1]);
    }
    /* Store the result vectors in proper order */
    ((__m128i *)dest)[k+0] = xmm1[0];
    ((__m128i *)dest)[k+1] = xmm1[8];
    ((__m128i *)dest)[k+2] = xmm1[4];
    ((__m128i *)dest)[k+3] = xmm1[12];
    ((__m128i *)dest)[k+4] = xmm1[2];
    ((__m128i *)dest)[k+5] = xmm1[10];
    ((__m128i *)dest)[k+6] = xmm1[6];
    ((__m128i *)dest)[k+7] = xmm1[14];
    ((__m128i *)dest)[k+8] = xmm1[1];
    ((__m128i *)dest)[k+9] = xmm1[9];
    ((__m128i *)dest)[k+10] = xmm1[5];
    ((__m128i *)dest)[k+11] = xmm1[13];
    ((__m128i *)dest)[k+12] = xmm1[3];
    ((__m128i *)dest)[k+13] = xmm1[11];
    ((__m128i *)dest)[k+14] = xmm1[7];
    ((__m128i *)dest)[k+15] = xmm1[15];
  }
}


/* Unshuffle a block.  This can never fail. */
void unshuffle(size_t bytesoftype, size_t blocksize,
               uint8_t* _src, uint8_t* _dest) {
  int unaligned_src = (int)((uintptr_t)_src % 16);
  int unaligned_dest = (int)((uintptr_t)_dest % 16);
  int power_of_two = (blocksize & (blocksize - 1)) == 0;
  int too_small = (blocksize < 256);

  if (unaligned_src || unaligned_dest || !power_of_two || too_small) {
    /* _src or _dest buffer is not aligned, not a power of two or is
       too small.  Call the non-sse2 version. */
    _unshuffle(bytesoftype, blocksize, _src, _dest);
    return;
  }

  /* Optimized unshuffle */
  /* The buffers must be aligned on a 16 bytes boundary, have a power */
  /* of 2 size and be larger or equal than 256 bytes. */
  if (bytesoftype == 4) {
    unshuffle4(_dest, _src, blocksize);
  }
  else if (bytesoftype == 8) {
    unshuffle8(_dest, _src, blocksize);
  }
  else if (bytesoftype == 16) {
    unshuffle16(_dest, _src, blocksize);
  }
  else if (bytesoftype == 2) {
    unshuffle2(_dest, _src, blocksize);
  }
  else {
    /* Non-optimized unshuffle */
    _unshuffle(bytesoftype, blocksize, _src, _dest);
  }
}

#else   /* no __SSE2__ available */

void shuffle(size_t bytesoftype, size_t blocksize,
             uint8_t* _src, uint8_t* _dest) {
  _shuffle(bytesoftype, blocksize, _src, _dest);
}

void unshuffle(size_t bytesoftype, size_t blocksize,
               uint8_t* _src, uint8_t* _dest) {
  _unshuffle(bytesoftype, blocksize, _src, _dest);
}

#endif  /* __SSE2__ */
