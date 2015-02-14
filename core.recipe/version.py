def main():
    inp = sys.stdin.read().strip().split('-')
    try:
        inp[1] = 'post%03d' % int(inp[1])
    except IndexError:
        pass
    print('.'.join(inp))
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
