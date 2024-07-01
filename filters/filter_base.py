from typing import Union

from pandas import DataFrame


class FilterBase(object):

    def __init__(self, column_id: Union[int, str], minn: int, maxx: int, name: str):
        self.cId = column_id
        self.minn = minn
        self.maxx = maxx
        self.name = name

    def __filter_cond(self, x) -> bool:
        # https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
        return (x > self.minn) & (x < self.maxx)

    def __call__(self, df: DataFrame) -> bool:
        return self.__filter_cond(df[self.cId])
