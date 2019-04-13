import time

def exe_time(func):
    """Record function running time. A decorator."""
    def newFunc(*args, **args2):
        t0 = time.time()
        # print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
        back = func(*args, **args2)
        # print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
        print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
        return back
    return newFunc

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_local_time():
  """return a string of local datatime"""
  return time.strftime("%y%m%d_%H%M%S", time.localtime())


def get_latest_folder(checkpoints_dir, nst_latest=1):
  """获取目录文件夹，根据创建时间排序，
  返回第 nst_latest 新的文件夹路径
  """
  files = os.listdir(checkpoints_dir)
  folders = []
  for file in files:
    path = os.path.join(checkpoints_dir, file)
    # logging.debug(path, os.path.getctime(path))
    if os.path.isdir(path):
      folders.append(path)
  folders.sort(key=lambda folder: os.path.getmtime(folder))
  return folders[-nst_latest]
