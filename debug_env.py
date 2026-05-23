import sys
import pkgutil

print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\nlangchain finder:", pkgutil.find_loader("langchain"))
print("langchain_core finder:", pkgutil.find_loader("langchain_core"))

try:
    import langchain
    print("\nlangchain module path:", langchain.__file__)
except Exception as e:
    print("\nError importing langchain:", repr(e))

try:
    import langchain_core
    print("langchain_core module path:", langchain_core.__file__)
except Exception as e:
    print("Error importing langchain_core:", repr(e))