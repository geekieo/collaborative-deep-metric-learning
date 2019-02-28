import sys
sys.path.append("..")
from utils import exe_time


def test_exe_time():
  def foo(a):
    return a**2
  a = 3
  res = exe_time(foo)(a)
  print(res)


if __name__ == "__main__":
    test_exe_time()