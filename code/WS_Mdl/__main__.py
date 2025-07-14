import sys
import WS_Mdl


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m WS_Mdl <function_name> <arguments>')
        sys.exit(1)

    function_name = sys.argv[1]
    args = sys.argv[2:]

    if hasattr(WS_Mdl, function_name):
        func = getattr(WS_Mdl, function_name)
        if callable(func):
            func(*args)
        else:
            print(f"Error: '{function_name}' is not callable.")
    else:
        print(f"Error: Function '{function_name}' not found in WS_Mdl.")


if __name__ == '__main__':
    main()
