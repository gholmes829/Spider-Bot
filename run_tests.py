from tests import edge_filter

def main():
    tests = [
        edge_filter.main,
    ]
    
    for test in tests:
        test()

if __name__ == '__main__':
    main()