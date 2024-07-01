import sys

from filters.filter_base import FilterBase


def create_filters_from_config(config):
    filters = []

    for f in config.extraction.filters:
        name = list(f.keys())[0]
        minn = -1
        if f[name]['min']:
            minn = f[name].min

        maxx = sys.maxsize
        if f[name]['max']:
            maxx = f[name].max

        columnId = f[name].columnId
        assert columnId is not None and columnId != ""

        filters.append(FilterBase(columnId, minn, maxx, name))

    return filters
