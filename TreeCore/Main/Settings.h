#pragma once
#define SUPPORT_RANGE_CODING
#define SUPPORT_HUFFMAN

#ifdef SUPPORT_HUFFMAN
	// this enabled the PDF to be changed in accordans to the Blackwords
	// #define ADJUST_PDF

	// for speed
	//#define USE_CUSTOM_ALLOC
	// msvcrt opimisation only use with USE_CUSTOM_ALLOC
	//#define MSVCRT_DEL_OPTIMIZE
	// #define USE_CUSTOM_ALLOC_CHECK

	// cant be 0.0078125*2 as that has max 0.0078125/2
	// cant be 0.0078125*4 as that is max 0.0078125
	// if not 1/pow(2,n) then min * 3 --> should be 3 else it should be 5 it is a aproximation anyway
	#define HUFF_MIN_BLACKOUT_MUL 5

	// #define DEBUG_TREE
#endif

#define PREALLOC_FACTOR_DECODE_FROM_BINARY 2
#define PREALLOC_FACTOR_decodeToBinary 2
// #define DEBUG_LOAD