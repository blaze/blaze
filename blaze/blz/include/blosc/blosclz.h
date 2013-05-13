/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted <faltet@gmail.com>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/

/*********************************************************************
  The code in this file is heavily based on FastLZ, a lightning-fast
  lossless compression library.  See LICENSES/FASTLZ.txt for details
  about copyright and rights to use.
**********************************************************************/


#ifndef BLOSCLZ_H
#define BLOSCLZ_H

#if defined (__cplusplus)
extern "C" {
#endif

/**
  Compress a block of data in the input buffer and returns the size of
  compressed block. The size of input buffer is specified by
  length. The minimum input buffer size is 16.

  The output buffer must be at least 5% larger than the input buffer
  and can not be smaller than 66 bytes.

  If the input is not compressible, or output does not fit in maxout
  bytes, the return value will be 0 and you will have to discard the
  output buffer.

  The input buffer and the output buffer can not overlap.
*/

int blosclz_compress(int opt_level, const void* input, int length,
                     void* output, int maxout);

/**
  Decompress a block of compressed data and returns the size of the
  decompressed block. If error occurs, e.g. the compressed data is
  corrupted or the output buffer is not large enough, then 0 (zero)
  will be returned instead.

  The input buffer and the output buffer can not overlap.

  Decompression is memory safe and guaranteed not to write the output buffer
  more than what is specified in maxout.
 */

int blosclz_decompress(const void* input, int length, void* output, int maxout);

#if defined (__cplusplus)
}
#endif

#endif /* BLOSCLZ_H */
