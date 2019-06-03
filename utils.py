# -*- coding:utf-8 -*-
import time
import os


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
  如果没有子目录，返回 checkpoints_dir
  """
  files = os.listdir(checkpoints_dir)
  folders = []
  for file in files:
    path = os.path.join(checkpoints_dir, file)
    # logging.debug(path, os.path.getctime(path))
    if os.path.isdir(path):
      folders.append(path)
  folders.sort(key=lambda folder: os.path.getmtime(folder))
  try:
    ckpt_dir = folders[-nst_latest]
  except:
    ckpt_dir = checkpoints_dir
    print('没有获得第 %d 新的文件夹路径，返回入参 %s'%(nst_latest,ckpt_dir))
  return ckpt_dir


def clean_file_by_time(log_dir, keepdays=7):
  """目录内部，文件将按时间无差别删除
  NOTE: 对于定时删除的目录，使用专属目录
  """
  for parent, dirnames, filenames in os.walk(log_dir):
    for filename in filenames:
      fullname = parent + "/" + filename  # 文件全称
      createTime = int(os.path.getctime(fullname))  # 文件创建时间
      nDayAgo = (datetime.datetime.now() -
                 datetime.timedelta(days=keepdays))  # 当前时间的n天前的时间
      timeStamp = int(time.mktime(nDayAgo.timetuple()))
      if createTime < timeStamp:  # 创建时间在n天前的文件删除
        os.remove(os.path.join(parent, filename))


def delete_file_by_count(base_dir, prefix, keepcount=7):
  pass