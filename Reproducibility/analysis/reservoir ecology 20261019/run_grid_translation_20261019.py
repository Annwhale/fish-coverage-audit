#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from concurrent.futures import ProcessPoolExecutor

from analyse_rimov import run_configuration, summarize

if __name__ == '__main__':
    # 审查指出固定原点未检验；这是主结果之后增加的探索性几何检查。
    settings = [('origin_e50', 100, 75, 5, 12, 'raw', 50, 0),
                ('origin_n50', 100, 75, 5, 12, 'raw', 0, 50),
                ('origin_en50', 100, 75, 5, 12, 'raw', 50, 50)]
    with ProcessPoolExecutor(max_workers=3) as executor:
        list(executor.map(run_configuration, settings))
    summarize()
