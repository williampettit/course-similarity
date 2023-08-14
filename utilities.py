import time

try:
  from termcolor import colored as col
except ImportError:
  col = lambda x, _: x


class ScopeTimer:
  def __init__(
    self,
    name: str,
    print_at_exit: bool = False,
  ) -> None:
    self.name = name
    self.print_at_exit = print_at_exit
    self.start = time.perf_counter()

  def _print_enter(self) -> None:
    print(col("[INFO]", "light_blue"), f"{self.name}...", end=" ")

  def __enter__(self) -> None:
    self._print_enter()
  
  def __exit__(self, exc_type, exc_value, traceback) -> None:
    self.end = time.perf_counter()
    
    if self.print_at_exit:
      self._print_enter()
    
    print(col(f"{self.end - self.start:.2f}s", "light_green"), end=" ")

    if exc_type is None:
      print(col("Good", "green"))
    else:
      print(col("Bad", "red"))
      print(exc_value)
      print(traceback)
