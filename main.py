def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def get_element(lst, index):
    if index < 0 or index >= len(lst):
        raise IndexError(f"Index {index} is out of range for list of length {len(lst)}")
    return lst[index]


def parse_number(value):
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid integer") from None


if __name__ == "__main__":
    # Division example
    print(divide(10, 2))

    # List access example
    items = [1, 2, 3]
    print(get_element(items, 1))

    # Parse number example
    print(parse_number("42"))
