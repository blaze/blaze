/*********************************************************************
  Blosc - Blocked Suffling and Compression Library

  Author: Francesc Alted <faltet@gmail.com>

  See LICENSES/BLOSC.txt for details about copyright and rights to use.
**********************************************************************/


/* Shuffle/unshuffle routines */

void shuffle(size_t bytesoftype, size_t blocksize,
             unsigned char* _src, unsigned char* _dest);

void unshuffle(size_t bytesoftype, size_t blocksize,
               unsigned char* _src, unsigned char* _dest);
