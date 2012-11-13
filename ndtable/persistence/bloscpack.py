#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim :set ft=py:

""" Command line interface to Blosc via python-blosc """

from __future__ import division

import sys
import os.path as path
import argparse
import struct
import math
import zlib
import hashlib
import itertools
from collections import OrderedDict
import blosc

__version__ = '0.3.0-dev'
__author__ = 'Valentin Haenel <valentin.haenel@gmx.de>'

EXTENSION = '.blp'
MAGIC = 'blpk'
BLOSCPACK_HEADER_LENGTH = 32
BLOSC_HEADER_LENGTH = 16
FORMAT_VERSION = 2
MAX_FORMAT_VERSION = 255
MAX_CHUNKS = (2**63)-1
DEFAULT_CHUNK_SIZE = '1M'
DEFAULT_OFFSETS = True
DEFAULT_OPTIONS = None  # created programatically later on
DEFAULT_TYPESIZE = 8
DEFAULT_CLEVEL = 7
DEAFAULT_SHUFFLE = True
BLOSC_ARGS = ['typesize', 'clevel', 'shuffle']
DEFAULT_BLOSC_ARGS = dict(zip(BLOSC_ARGS,
    (DEFAULT_TYPESIZE, DEFAULT_CLEVEL, DEAFAULT_SHUFFLE)))
NORMAL  = 'NORMAL'
VERBOSE = 'VERBOSE'
DEBUG   = 'DEBUG'
LEVEL = NORMAL
VERBOSITY_LEVELS = [NORMAL, VERBOSE, DEBUG]
PREFIX = "bloscpack.py"
SUFFIXES = OrderedDict((
             ("B", 2**0 ),
             ("K", 2**10),
             ("M", 2**20),
             ("G", 2**30),
             ("T", 2**40)))

class Hash(object):
    """ Uniform hash object.

    Parameters
    ----------
    name : str
        the name of the hash
    size : int
        the length of the digest in bytes
    function : callable
        the hash function implementation

    Notes
    -----
    The 'function' argument should return the raw bytes as string.

    """

    def __init__(self, name, size, function):
        self.name, self.size, self._function = name, size, function

    def __call__(self, data):
        return self._function(data)

def zlib_hash(func):
    """ Wrapper for zlib hashes. """
    def hash_(data):
        # The binary OR is recommended to obtain uniform hashes on all python
        # versions and platforms. The type with be 'uint32'.
        return struct.pack('<I', func(data) & 0xffffffff)
    return 4, hash_

def hashlib_hash(func):
    """ Wrapper for hashlib hashes. """
    def hash_(data):
        return func(data).digest()
    return func().digest_size, hash_

CHECKSUMS = [Hash('None', 0, lambda data: ''),
     Hash('adler32', *zlib_hash(zlib.adler32)),
     Hash('crc32', *zlib_hash(zlib.crc32)),
     Hash('md5', *hashlib_hash(hashlib.md5)),
     Hash('sha1', *hashlib_hash(hashlib.sha1)),
     Hash('sha224', *hashlib_hash(hashlib.sha224)),
     Hash('sha256', *hashlib_hash(hashlib.sha256)),
     Hash('sha384', *hashlib_hash(hashlib.sha384)),
     Hash('sha512', *hashlib_hash(hashlib.sha512)),
    ]
CHECKSUMS_AVAIL = [c.name for c in CHECKSUMS]
CHECKSUMS_LOOKUP = dict(((c.name, c) for c in CHECKSUMS))
DEFAULT_CHECKSUM = 'adler32'

def print_verbose(message, level=VERBOSE):
    """ Print message with desired verbosity level. """
    if level not in VERBOSITY_LEVELS:
        raise TypeError("Desired level '%s' is not one of %s" % (level,
            str(VERBOSITY_LEVELS)))
    if VERBOSITY_LEVELS.index(level) <= VERBOSITY_LEVELS.index(LEVEL):
        print('%s: %s' % (PREFIX, message))

def error(message, exit_code=1):
    """ Print message and exit with desired code. """
    for line in [l for l in message.split('\n') if l != '']:
        print('%s: error: %s' % (PREFIX, line))
    sys.exit(exit_code)

def pretty_size(size_in_bytes):
    """ Pretty print filesize.  """
    if size_in_bytes == 0:
        return "0B"
    for suf, lim in reversed(sorted(SUFFIXES.items(), key=lambda x: x[1])):
        if size_in_bytes < lim:
            continue
        else:
            return str(round(size_in_bytes/lim, 2))+suf

def double_pretty_size(size_in_bytes):
    """ Pretty print filesize including size in bytes. """
    return ("%s (%dB)" %(pretty_size(size_in_bytes), size_in_bytes))

def reverse_pretty(readable):
    """ Reverse pretty printed file size. """
    # otherwise we assume it has a suffix
    suffix = readable[-1]
    if suffix not in SUFFIXES.keys():
        raise ValueError(
                "'%s' is not a valid prefix multiplier, use one of: '%s'" %
                (suffix, SUFFIXES.keys()))
    else:
        return int(float(readable[:-1]) * SUFFIXES[suffix])

def decode_uint8(byte):
    return struct.unpack('<B', byte)[0]

def decode_uint32(fourbyte):
    return struct.unpack('<I', fourbyte)[0]

def decode_int32(fourbyte):
    return struct.unpack('<i', fourbyte)[0]

def decode_int64(eightbyte):
    return struct.unpack('<q', eightbyte)[0]

def decode_bitfield(byte):
    return bin(decode_uint8(byte))[2:].rjust(8,'0')

def encode_uint8(byte):
    return struct.pack('<B', byte)

def encode_int32(fourbyte):
    return struct.pack('<i', fourbyte)

def encode_int64(eightbyte):
    return struct.pack('<q', eightbyte)

class BloscPackCustomFormatter(argparse.HelpFormatter):
    """ Custom HelpFormatter.

    Basically a combination and extension of ArgumentDefaultsHelpFormatter and
    RawTextHelpFormatter. Adds default values to argument help, but only if the
    default is not in [None, True, False]. Also retains all whitespace as it
    is.

    """
    def _get_help_string(self, action):
        help_ = action.help
        if '%(default)' not in action.help \
                and action.default not in \
                [argparse.SUPPRESS, None, True, False]:
            defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
            if action.option_strings or action.nargs in defaulting_nargs:
                help_ += ' (default: %(default)s)'
        return help_

    def _split_lines(self, text, width):
        return text.splitlines()

def create_parser():
    """ Create and return the parser. """
    parser = argparse.ArgumentParser(
            #usage='%(prog)s [GLOBAL_OPTIONS] (compress | decompress)
            # [COMMAND_OPTIONS] <in_file> [<out_file>]',
            description='command line de/compression with blosc',
            formatter_class=BloscPackCustomFormatter)
    ## print version of bloscpack, python-blosc and blosc itself
    parser.add_argument('--version',
            action='version',
            version='%(prog)s:\t' + ("'%s'\n" % __version__) +
                    "python-blosc:\t'%s'\n"   % blosc.version.__version__ +
                    "blosc:\t\t'%s'\n"        % blosc.BLOSC_VERSION_STRING)
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('-v', '--verbose',
            action='store_true',
            default=False,
            help='be verbose about actions')
    output_group.add_argument('-d', '--debug',
            action='store_true',
            default=False,
            help='print debugging output too')
    global_group = parser.add_argument_group(title='global options')
    global_group.add_argument('-f', '--force',
            action='store_true',
            default=False,
            help='disable overwrite checks for existing files\n' +
            '(use with caution)')
    class CheckThreadOption(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if not 1 <= value <= blosc.BLOSC_MAX_THREADS:
                error('%s must be 1 <= n <= %d'
                        % (option_string, blosc.BLOSC_MAX_THREADS))
            setattr(namespace, self.dest, value)
    global_group.add_argument('-n', '--nthreads',
            metavar='[1, %d]' % blosc.BLOSC_MAX_THREADS,
            action=CheckThreadOption,
            default=blosc.ncores,
            type=int,
            dest='nthreads',
            help='set number of threads, (default: %(default)s (ncores))')

    subparsers = parser.add_subparsers(title='subcommands',
            metavar='', dest='subcommand')

    compress_parser = subparsers.add_parser('compress',
            formatter_class=BloscPackCustomFormatter,
            help='perform compression on file')
    c_parser = subparsers.add_parser('c',
            formatter_class=BloscPackCustomFormatter,
            help="alias for 'compress'")

    class CheckNchunksOption(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if not 1 <= value <= MAX_CHUNKS:
                error('%s must be 1 <= n <= %d'
                        % (option_string, MAX_CHUNKS))
            setattr(namespace, self.dest, value)
    class CheckChunkSizeOption(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            if value == 'max':
                value = blosc.BLOSC_MAX_BUFFERSIZE
            else:
                try:
                    # try to get the value as bytes
                    value = reverse_pretty(value)
                except ValueError as ve:
                    error('%s error: %s' % (option_string, ve.message +
                        " or 'max'"))
                if value < 0:
                    error('%s must be > 0 ' % option_string)
            setattr(namespace, self.dest, value)
    for p in [compress_parser, c_parser]:
        blosc_group = p.add_argument_group(title='blosc settings')
        blosc_group.add_argument('-t', '--typesize',
                metavar='<size>',
                default=DEFAULT_TYPESIZE,
                type=int,
                help='typesize for blosc')
        blosc_group.add_argument('-l', '--clevel',
                default=DEFAULT_CLEVEL,
                choices=range(10),
                metavar='[0, 9]',
                type=int,
                help='compression level')
        blosc_group.add_argument('-s', '--no-shuffle',
                action='store_false',
                default=DEAFAULT_SHUFFLE,
                dest='shuffle',
                help='deactivate shuffle')
        bloscpack_chunking_group = p.add_mutually_exclusive_group()
        bloscpack_chunking_group.add_argument('-c', '--nchunks',
                metavar='[1, 2**32-1]',
                action=CheckNchunksOption,
                type=int,
                default=None,
                help='set desired number of chunks')
        bloscpack_chunking_group.add_argument('-z', '--chunk-size',
                metavar='<size>',
                action=CheckChunkSizeOption,
                type=str,
                default=None,
                dest='chunk_size',
                help="set desired chunk size or 'max' (default: %s)" %
                DEFAULT_CHUNK_SIZE)
        bloscpack_group = p.add_argument_group(title='bloscpack settings')
        def join_with_eol(items):
            return ', '.join(items) + '\n'
        checksum_format = join_with_eol(CHECKSUMS_AVAIL[0:3]) + \
                join_with_eol(CHECKSUMS_AVAIL[3:6]) + \
                join_with_eol(CHECKSUMS_AVAIL[6:])
        checksum_help = 'set desired checksum:\n' + checksum_format
        bloscpack_group.add_argument('-k', '--checksum',
                metavar='<checksum>',
                type=str,
                choices=CHECKSUMS_AVAIL,
                default=DEFAULT_CHECKSUM,
                dest='checksum',
                help=checksum_help)
        bloscpack_group.add_argument('-o', '--no-offsets',
                action='store_false',
                default=DEFAULT_OFFSETS,
                dest='offsets',
                help='deactivate offsets')

    decompress_parser = subparsers.add_parser('decompress',
            formatter_class=BloscPackCustomFormatter,
            help='perform decompression on file')

    d_parser = subparsers.add_parser('d',
            formatter_class=BloscPackCustomFormatter,
            help="alias for 'decompress'")

    for p in [decompress_parser, d_parser]:
        p.add_argument('-e', '--no-check-extension',
                action='store_true',
                default=False,
                dest='no_check_extension',
                help='disable checking input file for extension (*.blp)\n' +
                '(requires use of <out_file>)')

    for p, help_in, help_out in [(compress_parser,
            'file to be compressed', 'file to compress to'),
                                 (c_parser,
            'file to be compressed', 'file to compress to'),
                                 (decompress_parser,
            'file to be decompressed', 'file to decompress to'),
                                 (d_parser,
            'file to be decompressed', 'file to decompress to'),
                                  ]:
        p.add_argument('in_file',
                metavar='<in_file>',
                type=str,
                default=None,
                help=help_in)
        p.add_argument('out_file',
                metavar='<out_file>',
                type=str,
                nargs='?',
                default=None,
                help=help_out)

    return parser

def decode_blosc_header(buffer_):
    """ Read and decode header from compressed Blosc buffer.

    Parameters
    ----------
    buffer_ : string of bytes
        the compressed buffer

    Returns
    -------
    settings : dict
        a dict containing the settings from Blosc

    Notes
    -----

    The Blosc 1.1.3 header is 16 bytes as follows::

        |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
        ^   ^   ^   ^ |     nbytes    |   blocksize   |    ctbytes    |
        |   |   |   |
        |   |   |   +--typesize
        |   |   +------flags
        |   +----------versionlz
        +--------------version

    The first four are simply bytes, the last three are are each unsigned ints
    (uint32) each occupying 4 bytes. The header is always little-endian.
    'ctbytes' is the length of the buffer including header and nbytes is the
    length of the data when uncompressed.

    """
    return {'version':   decode_uint8(buffer_[0]),
            'versionlz': decode_uint8(buffer_[1]),
            'flags':     decode_uint8(buffer_[2]),
            'typesize':  decode_uint8(buffer_[3]),
            'nbytes':    decode_uint32(buffer_[4:8]),
            'blocksize': decode_uint32(buffer_[8:12]),
            'ctbytes':   decode_uint32(buffer_[12:16])}

class ChunkingException(BaseException):
    pass

class NoSuchChecksum(ValueError):
    pass

class ChecksumMismatch(RuntimeError):
    pass

class FileNotFound(IOError):
    pass

def calculate_nchunks(in_file_size, nchunks=None, chunk_size=None):
    """ Determine chunking for an input file.

    Parameters
    ----------
    in_file_size : int
        the size of the input file
    nchunks : int, default: None
        the number of chunks desired by the user
    chunk_size : int, default: None
        the desired chunk size

    Returns
    -------
    nchunks, chunk_size, last_chunk_size

    nchunks : int
        the number of chunks
    chunk_size : int
        the size of each chunk in bytes
    last_chunk_size : int
        the size of the last chunk in bytes

    Raises
    ------
    ChunkingException
        under various error conditions

    Notes
    -----
    You must specify either 'nchunks' or 'chunk_size' but not neither or both.

    """
    if nchunks is not None and chunk_size is not None:
        raise ValueError(
                "either specify 'nchunks' or 'chunk_size', but not both")
    elif nchunks is None and chunk_size is None:
        raise ValueError(
                "you must specify either 'nchunks' or 'chunk_size'")
    elif in_file_size <= 0:
        raise ValueError(
                "'in_file_size' must be greater than zero")
    elif nchunks is not None and chunk_size is None:
        print_verbose("'nchunks' proposed", level=DEBUG)
        if nchunks > in_file_size:
            raise ChunkingException(
                    "Your value of 'nchunks': %d is " % nchunks +
                    "greater than the 'in_file size': %d" % in_file_size)
        elif nchunks <= 0:
            raise ChunkingException(
                    "'nchunks' must be greater than zero, not '%d' " % nchunks)
        quotient, remainder = divmod(in_file_size, nchunks)
        # WARNING: this is the most horrible piece of code in bloscpack
        # if you can do better, please, please send me patches
        # user wants a single chunk
        if nchunks == 1:
            chunk_size = 0
            last_chunk_size = in_file_size
        # perfect fit
        elif remainder == 0:
            chunk_size = quotient
            last_chunk_size = chunk_size
        # user wants two chunks
        elif nchunks == 2:
            chunk_size = quotient
            last_chunk_size = in_file_size - chunk_size
        # multiple chunks, if the nchunks is quite small, we may have a tiny
        # remainder and hence tiny last chunk
        else:
            chunk_size = in_file_size//(nchunks-1)
            last_chunk_size = in_file_size - chunk_size * (nchunks-1)
    elif nchunks is None and chunk_size is not None:
        print_verbose("'chunk_size' proposed", level=DEBUG)
        if chunk_size > in_file_size:
            raise ChunkingException(
                    "Your value of 'chunk_size': %d is " % chunk_size +
                    "greater than the 'in_file size': %d" % in_file_size)
        elif chunk_size <= 0:
            raise ChunkingException(
                    "'chunk_size' must be greater than zero, not '%d' " %
                    chunk_size)
        quotient, remainder = divmod(in_file_size, chunk_size)
        # the user wants a single chunk
        if chunk_size == in_file_size:
            nchunks = 1
            chunk_size = 0
            last_chunk_size = in_file_size
        # no remainder, perfect fit
        elif remainder == 0:
            nchunks = quotient
            last_chunk_size = chunk_size
        # with a remainder
        else:
            nchunks = quotient + 1
            last_chunk_size = remainder
    if chunk_size > blosc.BLOSC_MAX_BUFFERSIZE \
            or last_chunk_size > blosc.BLOSC_MAX_BUFFERSIZE:
        raise ChunkingException(
            "Your value of 'nchunks' would lead to chunk sizes bigger than " +
            "'BLOSC_MAX_BUFFERSIZE', please use something smaller.\n" +
            "nchunks : %d\n" % nchunks +
            "chunk_size : %d\n" % chunk_size +
            "last_chunk_size : %d\n" % last_chunk_size +
            "BLOSC_MAX_BUFFERSIZE : %d\n" % blosc.BLOSC_MAX_BUFFERSIZE)
    elif nchunks > MAX_CHUNKS:
        raise ChunkingException(
                "nchunks: '%d' is greater than the MAX_CHUNKS: '%d'" %
                (nchunks, MAX_CHUNKS))
    print_verbose('nchunks: %d' % nchunks, level=VERBOSE)
    print_verbose('chunk_size: %s' % double_pretty_size(chunk_size),
            level=VERBOSE)
    print_verbose('last_chunk_size: %s' % double_pretty_size(last_chunk_size),
            level=DEBUG)
    return nchunks, chunk_size, last_chunk_size

def check_range(name, value, min_, max_):
    """ Check that a variable is in range. """
    if not isinstance(value, (int, long)):
        raise TypeError("'%s' must be of type 'int'" % name)
    elif not min_ <= value <= max_:
        raise ValueError(
                "'%s' must be in the range %s <= n <= %s, not '%s'" %
                tuple(map(str, (name, min, max_, value))))

def _check_options(options):
    """ Check the options bitfield.

    Parameters
    ----------
    options : str

    Raises
    ------
    TypeError
        if options is not a string
    ValueError
        either if any character in option is not a zero or a one, or if options
        is not of length 8
    """

    if not isinstance(options, str):
        raise TypeError("'options' must be of type 'str', not '%s'" %
                type(options))
    elif (not len(options) == 8 or
            not all(map(lambda x: x in ['0', '1'], iter(options)))):
        raise ValueError(
                "'options' must be string of 0s and 1s of length 8, not '%s'" %
                options)

def create_options(offsets=DEFAULT_OFFSETS):
    """ Create the options bitfield.

    Parameters
    ----------
    offsets : bool
    """
    return "".join([str(int(i)) for i in
        [False, False, False, False, False, False, False, offsets]])

def decode_options(options):
    """ Parse the options bitfield.

    Parameters
    ----------
    options : str
        the options bitfield

    Returns
    -------
    options : dict mapping str -> bool
    """

    _check_options(options)
    return {'offsets': bool(int(options[7]))}

# default options created here programatically
DEFAULT_OPTIONS = create_options()


def create_bloscpack_header(format_version=FORMAT_VERSION,
        options='00000000',
        checksum=0,
        typesize=0,
        chunk_size=-1,
        last_chunk=-1,
        nchunks=-1):
    """ Create the bloscpack header string.

    Parameters
    ----------
    format_version : int
        the version format for the compressed file
    options : bitfield (string of 0s and 1s)
        the options for this file
    checksum : int
        the checksum to be used
    typesize : int
        the typesize used for blosc in the chunks
    chunk_size : int
        the size of a regular chunk
    last_chunk : int
        the size of the last chunk
    nchunks : int
        the number of chunks

    Returns
    -------
    bloscpack_header : string
        the header as string

    Notes
    -----

    See the file 'header_rfc.rst' distributed with the source code for
    details on the header format.

    Raises
    ------
    ValueError
        if any of the arguments have an invalid value
    TypeError
        if any of the arguments have the wrong type

    """
    check_range('format_version', format_version, 0, MAX_FORMAT_VERSION)
    _check_options(options)
    check_range('checksum',   checksum, 0, len(CHECKSUMS))
    check_range('typesize',   typesize,    0, blosc.BLOSC_MAX_TYPESIZE)
    check_range('chunk_size', chunk_size, -1, blosc.BLOSC_MAX_BUFFERSIZE)
    check_range('last_chunk', last_chunk, -1, blosc.BLOSC_MAX_BUFFERSIZE)
    check_range('nchunks',    nchunks,    -1, MAX_CHUNKS)

    format_version = encode_uint8(format_version)
    options = encode_uint8(int(options, 2))
    checksum = encode_uint8(checksum)
    typesize = encode_uint8(typesize)
    chunk_size = encode_int32(chunk_size)
    last_chunk = encode_int32(last_chunk)
    nchunks = encode_int64(nchunks)
    RESERVED = encode_int64(0)

    return (MAGIC + format_version + options + checksum + typesize +
            chunk_size + last_chunk +
            nchunks +
            RESERVED)

def decode_bloscpack_header(buffer_):
    """ Check that the magic marker exists and return number of chunks.

    Parameters
    ----------
    buffer_ : str of length 32 (but probably any sequence would work)
        the header

    Returns
    -------
    format_version : int
        the version format for the compressed file
    options : bitfield (string of 0s and 1s)
        the options for this file
    checksum : int
        the checksum to be used
    typesize : int
        the typesize used for blosc in the chunks
    chunk_size : int
        the size of a regular chunk
    last_chunk : int
        the size of the last chunk
    nchunks : int
        the number of chunks
    RESERVED : int
        the RESERVED field from the header, should be zero

    """
    if len(buffer_) != 32:
        raise ValueError(
            "attempting to decode a bloscpack header of length '%d', not '32'"
            % len(buffer_))
    elif buffer_[0:4] != MAGIC:
        raise ValueError(
            "the magic marker '%s' is missing from the bloscpack " % MAGIC +
            "header, instead we found: '%s'" % buffer_[0:4])

    return {'format_version': decode_uint8(buffer_[4]),
            'options':        decode_bitfield(buffer_[5]),
            'checksum':       decode_uint8(buffer_[6]),
            'typesize':       decode_uint8(buffer_[7]),
            'chunk_size':     decode_int32(buffer_[8:12]),
            'last_chunk':     decode_int32(buffer_[12:16]),
            'nchunks':        decode_int64(buffer_[16:24]),
            'RESERVED':       decode_int64(buffer_[24:32]),
            }

def process_compression_args(args):
    """ Extract and check the compression args after parsing by argparse.

    Parameters
    ----------
    args : argparse.Namespace
        the parsed command line arguments

    Returns
    -------
    in_file : str
        the input file name
    out_file : str
        the out_file name
    blosc_args : tuple of (int, int, bool)
        typesize, clevel and shuffle
    """
    in_file = args.in_file
    out_file = in_file + EXTENSION \
        if args.out_file is None else args.out_file
    blosc_args = dict((arg, args.__getattribute__(arg)) for arg in BLOSC_ARGS)
    return in_file, out_file, blosc_args

def process_decompression_args(args):
    """ Extract and check the decompression args after parsing by argparse.

    Warning: may call sys.exit()

    Parameters
    ----------
    args : argparse.Namespace
        the parsed command line arguments

    Returns
    -------
    in_file : str
        the input file name
    out_file : str
        the out_file name
    """
    in_file = args.in_file
    out_file = args.out_file
    # remove the extension for output file
    if args.no_check_extension:
        if out_file is None:
            error('--no-check-extension requires use of <out_file>')
    else:
        if in_file.endswith(EXTENSION):
            out_file = in_file[:-len(EXTENSION)] \
                    if args.out_file is None else args.out_file
        else:
            error("input file '%s' does not end with '%s'" %
                    (in_file, EXTENSION))
    return in_file, out_file

def check_files(in_file, out_file, args):
    """ Check files exist/don't exist.

    Parameters
    ----------
    in_file : str:
        the input file
    out_file : str
        the output file
    args : parser args
        any additional arguments from the parser

    Raises
    ------
    FileNotFound
        in case any of the files isn't found.

    """
    if not path.exists(in_file):
        raise FileNotFound("input file '%s' does not exist!" % in_file)
    if path.exists(out_file):
        if not args.force:
            raise FileNotFound("output file '%s' exists!" % out_file)
        else:
            print_verbose("overwriting existing file: %s" % out_file)
    print_verbose('input file is: %s' % in_file)
    print_verbose('output file is: %s' % out_file)

def process_nthread_arg(args):
    """ Extract and set nthreads. """
    if args.nthreads != blosc.ncores:
        blosc.set_nthreads(args.nthreads)
    print_verbose('using %d thread%s' %
            (args.nthreads, 's' if args.nthreads > 1 else ''))

def pack_file(in_file, out_file, blosc_args, nchunks=None, chunk_size=None,
        offsets=DEFAULT_OFFSETS, checksum=DEFAULT_CHECKSUM):
    """ Main function for compressing a file.

    Parameters
    ----------
    in_file : str
        the name of the input file
    out_file : str
        the name of the output file
    blosc_args : dict
        dictionary of blosc keyword args
    nchunks : int, default: None
        The desired number of chunks.
    chunk_size : int, default: None
        The desired chunk size in bytes.
    offsets : bool
        Wheather to include offsets.
    checksum : str
        Which checksum to use.

    Notes
    -----
    The parameters 'nchunks' and 'chunk_size' are mutually exclusive. Will be
    determined automatically if not present.

    """
    # calculate chunk sizes
    in_file_size = path.getsize(in_file)
    print_verbose('input file size: %s' % double_pretty_size(in_file_size))
    nchunks, chunk_size, last_chunk_size = \
            calculate_nchunks(in_file_size, nchunks, chunk_size)
    # calculate header
    options = create_options(offsets=offsets)
    if offsets:
        offsets_storage = list(itertools.repeat(0, nchunks))
    # set the checksum impl
    checksum_impl = CHECKSUMS_LOOKUP[checksum]
    raw_bloscpack_header = create_bloscpack_header(
            options=options,
            checksum=CHECKSUMS_AVAIL.index(checksum),
            typesize=blosc_args['typesize'],
            chunk_size=chunk_size,
            last_chunk=last_chunk_size,
            nchunks=nchunks
            )
    print_verbose('raw_bloscpack_header: %s' % repr(raw_bloscpack_header),
            level=DEBUG)
    # write the chunks to the file
    with open(in_file, 'rb') as input_fp, \
         open(out_file, 'wb') as output_fp:
        output_fp.write(raw_bloscpack_header)
        # preallocate space for the offsets
        if offsets:
            output_fp.write(encode_int64(-1) * nchunks)
        # if nchunks == 1 the last_chunk_size is the size of the single chunk
        for i, bytes_to_read in enumerate((
                [chunk_size] * (nchunks - 1)) + [last_chunk_size]):
            # store the current position in the file
            if offsets:
                offsets_storage[i] = output_fp.tell()
            current_chunk = input_fp.read(bytes_to_read)
            # do compression
            compressed = blosc.compress(current_chunk, **blosc_args)
            # write compressed data
            output_fp.write(compressed)
            print_verbose("chunk '%d'%s written, in: %s out: %s ratio: %s" %
                    (i, ' (last)' if i == nchunks - 1 else '',
                    double_pretty_size(len(current_chunk)),
                    double_pretty_size(len(compressed)),
                    "%0.3f" % (len(compressed) / len(current_chunk))
                    if len(current_chunk) != 0 else "N/A"),
                    level=DEBUG)
            tail_mess = ""
            if checksum_impl.size > 0:
                # compute the checksum on the compressed data
                digest = checksum_impl(compressed)
                # write digest
                output_fp.write(digest)
                tail_mess += ('checksum (%s): %s ' % (checksum, repr(digest)))
            if offsets:
                tail_mess += ("offset: '%d'" % offsets_storage[i])
            if len(tail_mess) > 0:
                print_verbose(tail_mess, level=DEBUG)
        if offsets:
            # seek to 32 bits into the file
            output_fp.seek(BLOSCPACK_HEADER_LENGTH, 0)
            print_verbose("Writing '%d' offsets: '%s'" %
                    (len(offsets_storage), repr(offsets_storage)), level=DEBUG)
            # write the offsets encoded into the reserved space in the file
            encoded_offsets = "".join([encode_int64(i) for i in offsets_storage])
            print_verbose("Raw offsets: %s" % repr(encoded_offsets),
                    level=DEBUG)
            output_fp.write(encoded_offsets)
    out_file_size = path.getsize(out_file)
    print_verbose('output file size: %s' % double_pretty_size(out_file_size))
    print_verbose('compression ratio: %f' % (out_file_size/in_file_size))

def unpack_file(in_file, out_file):
    """ Main function for decompressing a file.

    Parameters
    ----------
    in_file : str
        the name of the input file
    out_file : str
        the name of the output file
    """
    in_file_size = path.getsize(in_file)
    print_verbose('input file size: %s' % pretty_size(in_file_size))
    with open(in_file, 'rb') as input_fp, \
         open(out_file, 'wb') as output_fp:
        # read the bloscpack header
        print_verbose('reading bloscpack header', level=DEBUG)
        bloscpack_header_raw = input_fp.read(BLOSCPACK_HEADER_LENGTH)
        print_verbose('bloscpack_header_raw: %s' %
                repr(bloscpack_header_raw), level=DEBUG)
        bloscpack_header = decode_bloscpack_header(bloscpack_header_raw)
        for arg, value in bloscpack_header.iteritems():
            # hack the values of the bloscpack header into the namespace
            globals()[arg] = value
            print_verbose('\t%s: %s' % (arg, value), level=DEBUG)
        checksum_impl = CHECKSUMS[checksum]
        if FORMAT_VERSION != format_version:
            error("format version of file was not '%s' as expected, but '%d'" %
                    (FORMAT_VERSION, format_version))
        # read the offsets
        options = decode_options(bloscpack_header['options'])
        if options['offsets']:
            offsets_raw = input_fp.read(8 * nchunks)
            print_verbose('Read raw offsets: %s' % repr(offsets_raw),
                    level=DEBUG)
            offset_storage = [decode_int64(offsets_raw[j-8:j]) for j in
                    xrange(8, nchunks*8+1, 8)]
            print_verbose('Offsets: %s' % offset_storage, level=DEBUG)
        # decompress
        for i in range(nchunks):
            print_verbose("decompressing chunk '%d'%s" %
                    (i, ' (last)' if i == nchunks-1 else ''), level=DEBUG)
            # read blosc header
            blosc_header_raw = input_fp.read(BLOSC_HEADER_LENGTH)
            blosc_header = decode_blosc_header(blosc_header_raw)
            print_verbose('blosc_header: %s' % repr(blosc_header), level=DEBUG)
            ctbytes = blosc_header['ctbytes']
            # Seek back BLOSC_HEADER_LENGTH bytes in file relative to current
            # position. Blosc needs the header too and presumably this is
            # better than to read the whole buffer and then concatenate it...
            input_fp.seek(-BLOSC_HEADER_LENGTH, 1)
            # read chunk
            compressed = input_fp.read(ctbytes)
            if checksum_impl.size > 0:
                # do checksum
                expected_digest = input_fp.read(checksum_impl.size)
                received_digest = checksum_impl(compressed)
                if received_digest != expected_digest:
                    raise ChecksumMismatch(
                            "Checksum mismatch detected in chunk '%d' " % i +
                            "expected: '%s', received: '%s'" %
                            (repr(expected_digest), repr(received_digest)))
                else:
                    print_verbose('checksum OK (%s): %s ' %
                            (checksum_impl.name, repr(received_digest)),
                            level=DEBUG)
            # if checksum OK, decompress buffer
            decompressed = blosc.decompress(compressed)
            # write decompressed chunk
            output_fp.write(decompressed)
            print_verbose("chunk written, in: %s out: %s" %
                    (pretty_size(len(compressed)),
                        pretty_size(len(decompressed))), level=DEBUG)
    out_file_size = path.getsize(out_file)
    print_verbose('output file size: %s' % pretty_size(out_file_size))
    print_verbose('decompression ratio: %f' % (out_file_size/in_file_size))

if __name__ == '__main__':
    parser = create_parser()
    PREFIX = parser.prog
    args = parser.parse_args()
    if args.verbose:
        LEVEL = VERBOSE
    elif args.debug:
        LEVEL = DEBUG
    print_verbose('command line argument parsing complete', level=DEBUG)
    print_verbose('command line arguments are: ', level=DEBUG)
    for arg, val in vars(args).iteritems():
        if arg == 'chunk_size' and val is not None:
            print_verbose('\t%s: %s' % (arg, double_pretty_size(val)),
                    level=DEBUG)
        else:
            print_verbose('\t%s: %s' % (arg, str(val)), level=DEBUG)

    # compression and decompression handled via subparsers
    if args.subcommand in ['compress', 'c']:
        print_verbose('getting ready for compression')
        in_file, out_file, blosc_args = process_compression_args(args)
        print_verbose('blosc args are:', level=DEBUG)
        for arg, value in blosc_args.iteritems():
            print_verbose('\t%s: %s' % (arg, value), level=DEBUG)
        try:
            check_files(in_file, out_file, args)
        except FileNotFound as fnf:
            error(str(fnf))
        process_nthread_arg(args)
        # mutually exclusivity in parser protects us from both having a value
        if args.nchunks is None and args.chunk_size is None:
            # file is larger than the default size... use it
            in_file_size = path.getsize(in_file)
            if in_file_size > reverse_pretty(DEFAULT_CHUNK_SIZE):
                args.chunk_size = reverse_pretty(DEFAULT_CHUNK_SIZE)
                print_verbose("Using default chunk-size: '%s'" %
                        DEFAULT_CHUNK_SIZE, level=DEBUG)
            # file is smaller than the default size, make a single chunk
            else:
                args.nchunks = 1
                print_verbose("File was smaller than the default " +
                        "chunk-size, using a single chunk")
        try:
            pack_file(in_file, out_file, blosc_args,
                    nchunks=args.nchunks,
                    chunk_size=args.chunk_size,
                    offsets=args.offsets,
                    checksum=args.checksum)
        except ChunkingException as e:
            error(e.message)
    elif args.subcommand in ['decompress', 'd']:
        print_verbose('getting ready for decompression')
        in_file, out_file = process_decompression_args(args)
        try:
            check_files(in_file, out_file, args)
        except FileNotFound as fnf:
            error(str(fnf))
        process_nthread_arg(args)
        try:
            unpack_file(in_file, out_file)
        except ValueError as ve:
            error(ve.message)
    else:
        # we should never reach this
        error('You found the easter-egg, please contact the author')
    print_verbose('done')
