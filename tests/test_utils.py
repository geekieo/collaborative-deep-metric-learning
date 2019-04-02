import sys
sys.path.append("..")
import utils
import test_utils_class

def test_exe_time():
  def foo(a):
    return a**2
  a = 3
  res = utils.exe_time(foo)(a)
  print(res)

def test_find_class_by_name():
  b = utils.find_class_by_name("test_class_B", [test_utils_class])()
  assert type(b).__name__ == "test_class_B"
  # print(type(b).__name__)

def test_get_local_time():
  print(utils.get_local_time())


if __name__ == "__main__":
  # test_exe_time()
  # test_find_class_by_name()
  test_get_local_time()