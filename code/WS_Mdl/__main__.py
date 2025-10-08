import sys


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m WS_Mdl <function_name> <arguments>')
        sys.exit(1)

    function_name = sys.argv[1]
    args = sys.argv[2:]

    # Lazy import - only import modules when needed
    modules_to_check = ['utils', 'geo', 'utils_imod']

    for module_name in modules_to_check:
        try:
            module = __import__(f'WS_Mdl.{module_name}', fromlist=[function_name])
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                if callable(func):
                    func(*args)
                    return
                else:
                    print(f"Error: '{function_name}' is not callable.")
                    return
        except ImportError:
            continue

    print(f"Error: Function '{function_name}' not found in any WS_Mdl modules.")


if __name__ == '__main__':
    main()
