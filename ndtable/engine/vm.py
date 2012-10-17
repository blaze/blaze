# ======================
# Toy Array VM in Python
# ======================

#  +--------+-----------+-----------+---+-----------+
#  | opcode | name      | assembler | A | stack     |
#  +========+===========+===========+===+===========+
#  |  0     | NOP       | nop,      |   | ``-``     |
#  +--------+-----------+-----------+---+-----------+
#  |  1     | LIT       | lit,      | x | ``-n``    |
#  +--------+-----------+-----------+---+-----------+
#  |  2     | DUP       | dup,      |   | ``n-nn``  |
#  +--------+-----------+-----------+---+-----------+
#  |  3     | DROP      | drop,     |   | ``n-``    |
#  +--------+-----------+-----------+---+-----------+
#  |  4     | SWAP      | swap,     |   | ``xy-yx`` |
#  +--------+-----------+-----------+---+-----------+
#  |  5     | PUSH      | push,     |   | ``n-``    |
#  +--------+-----------+-----------+---+-----------+
#  |  6     | POP       | pop,      |   | ``-n``    |
#  +--------+-----------+-----------+---+-----------+
#  |  7     | LOOP      | loop,     | x | ``n-n``   |
#  +--------+-----------+-----------+---+-----------+
#  |  8     | JUMP      | jump,     | x | ``-``     |
#  +--------+-----------+-----------+---+-----------+
#  |  9     | RETURN    | ;,        |   | ``-``     |
#  +--------+-----------+-----------+---+-----------+
#  | 10     | LT_JUMP   | <jump,    | x | ``xy-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 11     | GT_JUMP   | >jump,    | x | ``xy-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 12     | NE_JUMP   | !jump,    | x | ``xy-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 13     | EQ_JUMP   | =jump,    | x | ``xy-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 14     | FETCH     | @,        |   | ``a-n``   |
#  +--------+-----------+-----------+---+-----------+
#  | 15     | STORE     | !,        |   | ``na-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 16     | ADD       | +,        |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 17     | SUBTRACT  | -,        |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 18     | MULTIPLY  | ``*``,    |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 19     | DIVMOD    | /mod,     |   | ``xy-rq`` |
#  +--------+-----------+-----------+---+-----------+
#  | 20     | AND       | and,      |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 21     | OR        | or,       |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 22     | XOR       | xor,      |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 23     | SHL       | <<,       |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 24     | SHR       | >>,       |   | ``xy-z``  |
#  +--------+-----------+-----------+---+-----------+
#  | 25     | ZERO_EXIT | 0;        |   | ``n-?``   |
#  +--------+-----------+-----------+---+-----------+
#  | 26     | INC       | 1+,       |   | ``x-y``   |
#  +--------+-----------+-----------+---+-----------+
#  | 27     | DEC       | 1-,       |   | ``x-y``   |
#  +--------+-----------+-----------+---+-----------+
#  | 28     | IN        | in,       |   | ``p-n``   |
#  +--------+-----------+-----------+---+-----------+
#  | 29     | OUT       | out,      |   | ``np-``   |
#  +--------+-----------+-----------+---+-----------+
#  | 30     | EXIT      |           |   |           |
#  +--------+-----------+-----------+---+-----------+
#  | 31     | VEC       |           |   |           |
#  +--------+-----------+-----------+---+-----------+

from struct import pack, unpack

EXIT = 0x0FFFFFFF

# -----------------------------------------------------------------------------
def mod( a, b ):
    x = abs(a)
    y = abs(b)
    q, r = divmod(x, y)

    if a < 0 and b < 0:
        r *= -1
    elif a > 0 and b < 0:
        q *= -1
    elif a < 0 and b > 0:
        r *= -1
        q *= -1

    return q, r

# -----------------------------------------------------------------------------
def vm( memory, descriptors ):

    # Pointer
    # =======
    ip = 0

    # Stacks
    # ======
    stack   = [] * 128
    address = [] * 1024
    index   = [] * 1024

    # Inputs
    # ======
    ports   = descriptors

    while ip < EXIT:
        try:
            opcode = memory[ip]
        except IndexError:
            raise RuntimeError('Overflow')

        print opcode, ip

        if opcode > 30:
            address.append( ip )
            ip = memory[ip] - 1

            while memory[ip + 1] == 0:
                ip += 1

        else:
            if  opcode  ==    0:     # nop
                pass

            elif opcode ==    1:     # lit
                ip += 1
                stack.append( memory[ip] )

            elif opcode ==    2:     # dup
                stack.append( stack[-1] )

            elif opcode ==    3:     # drop
                stack.pop()

            elif opcode ==    4:     # swap
                a = stack[-2]
                stack[-2] = stack[-1]
                stack[-1] = a

            elif opcode ==    5:     # push
                address.append( stack.pop() )

            elif opcode ==    6:     # pop
                stack.append( address.pop() )

            elif opcode ==    7:     # loop
                stack[-1] -= 1
                if stack[-1] != 0 and stack[-1] > -1:
                    ip += 1
                    ip = memory[ip] - 1
                else:
                    ip += 1
                    stack.pop()

            elif opcode ==    8:     # jump
                ip += 1
                ip = memory[ip] - 1
                if memory[ip + 1] == 0:
                    ip += 1
                    if memory[ip + 1] == 0:
                        ip += 1

            elif opcode ==    9:     # return
                ip = address.pop()
                if memory[ip + 1] == 0:
                    ip += 1
                    if memory[ip + 1] == 0:
                        ip += 1

            elif opcode == 10:     # >= jump
                ip += 1
                a = stack.pop()
                b = stack.pop()
                if b > a:
                    ip = memory[ip] - 1

            elif opcode == 11:     # <= jump
                ip += 1
                a = stack.pop()
                b = stack.pop()
                if b < a:
                    ip = memory[ip] - 1

            elif opcode == 12:     # != jump
                ip += 1
                a = stack.pop()
                b = stack.pop()
                if b != a:
                    ip = memory[ip] - 1

            elif opcode == 13:     # == jump
                ip += 1
                a = stack.pop()
                b = stack.pop()
                if b == a:
                    ip = memory[ip] - 1

            elif opcode == 14:     # @
                stack[-1] = memory[stack[-1]]

            elif opcode == 15:     # !
                mi = stack.pop()
                memory[mi] = stack.pop()

            elif opcode == 16:     # +
                t = stack.pop()
                stack[ -1 ] += t
                stack[-1] = unpack('=l', pack('=L', stack[-1] & 0xffffffff))[0]

            elif opcode == 17:     # -
                t = stack.pop()
                stack[-1] -= t
                stack[-1] = unpack('=l', pack('=L', stack[-1] & 0xffffffff))[0]

            elif opcode == 18:     # *
                t = stack.pop()
                stack[-1] *= t
                stack[-1] = unpack('=l', pack('=L', stack[-1] & 0xffffffff))[0]

            elif opcode == 19:     # /mod
                a = stack[-1]
                b = stack[-2]
                stack[-1], stack[-2] = mod( b, a )
                stack[-1] = unpack('=l', pack('=L', stack[-1] & 0xffffffff))[0]
                stack[-2] = unpack('=l', pack('=L', stack[-2] & 0xffffffff))[0]

            elif opcode == 20:     # and
                t = stack.pop()
                stack[-1] &= t

            elif opcode == 21:     # or
                t = stack.pop()
                stack[-1] |= t

            elif opcode == 22:     # xor
                t = stack.pop()
                stack[-1] ^= t

            elif opcode == 23:     # <<
                t = stack.pop()
                stack[-1] <<= t

            elif opcode == 24:     # >>
                t = stack.pop()
                stack[-1] >>= t

            elif opcode == 25:     # 0;
                if stack[-1] == 0:
                    stack.pop()
                    ip = address.pop()

            elif opcode == 26:     # inc
                stack[-1] += 1

            elif opcode == 27:     # dec
                stack[-1] -= 1

            elif opcode == 28:     # in
                t = stack[-1]
                stack[-1] = ports[t]
                ports[t] = 0

            elif opcode == 29:     # out
                pi = stack.pop()
                ports[ pi ] = stack.pop()

            elif opcode == 30:     # exit
                ip = EXIT

            elif opcode == 31:     # vec
                mi  = stack.pop()
                idx = index.pop()

                for i in xrange(idx):
                    stack.append( memory[mi+i] )

        ip += 1

# -----------------------------------------------------------------------------
def run():
    memory = [0] * 100
    descriptors = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vm( memory, descriptors )

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run()
