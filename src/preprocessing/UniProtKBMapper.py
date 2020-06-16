import urllib.parse
import urllib.request
import numpy as np
import pandas as pd
from typing import List, Dict
import csv
import math

class UniProtKBMapper:

    def __init__(self, data: pd.DataFrame, id_mappings: List[str]) -> None:
        self.data = data.values
        self.nodes = self.get_nodes(self.data)
        self.mapping = self.request_multiple_mappings(id_mappings)
        self.converted = self.convert()
        self.converted_nodes = self.get_nodes(self.converted)

    def get_nodes(self, nparray: np.array) -> np.array:
        return np.unique(nparray.flatten())

    def request_multiple_mappings(self, id_mappings: List[str]) -> Dict[str, str]:

        mapping = {}
        for id_mapping in id_mappings:
            per_batch = 1000
            total_batches = math.ceil(len(self.nodes)/per_batch)
            for i in range(0, total_batches):
                nodes = self.nodes[i*per_batch:(i+1)*per_batch]

                print('Downloading {} mapping ... {}/{}'.format(id_mapping, i+1, total_batches))

                temp_mapping = self.request_mapping(nodes, id_mapping)
                mapping = {**mapping, **temp_mapping}

        return mapping

    def request_mapping(self, nodes: np.array, id_mapping: str) -> Dict[str, str]:

        mapping_dict = {}
        url = 'https://www.uniprot.org/uploadlists/'

        params = {
        'from': id_mapping,
        'to': 'ACC',
        'format': 'tab',
        'query': ' '.join(nodes)
        }

        data = urllib.parse.urlencode(params)
        data = data.encode('utf-8')
        req = urllib.request.Request(url, data)

        with urllib.request.urlopen(req) as f:
            next(f)
            for line in f.readlines():
                mapping = line.decode('utf-8').split()
                mapping_dict[str(mapping[0])] = str(mapping[1])

        return mapping_dict

    def convert(self) -> np.array:

        converted_data = []
        for edge in self.data:
            if all(x in self.mapping for x in edge):
                map_one = self.mapping[edge[0]]
                map_two = self.mapping[edge[1]]
                sorted_values = sorted([map_one, map_two])
                converted_data.append(tuple(sorted_values))

        unique_converted_data = np.array([list(x) for x in set(converted_data)])
        return unique_converted_data

    def get_converted(self) -> pd.DataFrame:
        return pd.DataFrame(self.converted)

    def save_mapping(self, output: str) -> None:
        with open(output, 'w') as f:
            for key in self.mapping:
                f.write("%s,%s\n"%(key,self.mapping[key]))
