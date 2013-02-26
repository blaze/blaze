def runner():
    import nose
    argv = ['-s', 'blaze']
    nose.run(argv=argv)

if __name__ == '__main__':
    runner()
