"""核对演练资料的公开输入约定。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inspect_packet import read_packet


class PacketTest(unittest.TestCase):
    def test_related_records(self):
        doc = read_packet(Path("资料/演练.json"))
        self.assertTrue({(row['session'],row['code']) for row in doc['scores']} == {(row['session'],row['code']) for row in doc['blind_map']})
        samples={row['sample'] for row in doc['samples']}
        self.assertTrue({row['sample'] for row in doc['blind_map']} <= samples)
        self.assertTrue({(row['session'],row['sample']) for row in doc['invalidations']} <= {(row['session'],row['sample']) for row in doc['blind_map']})
        self.assertTrue(all(set(row['anchors']+row['items']) <= samples for row in doc['label_requests']))
        self.assertTrue(all(0<=row[key]<=10 for row in doc['scores'] for key in ('aroma','crisp')))
        self.assertTrue({row['id'] for row in doc['labels']} <= {row['id'] for row in doc['label_requests']})

    def test_reject_wrong_container(self):
        with TemporaryDirectory() as folder:
            packet = Path(folder) / "unexpected.json"
            packet.write_text(json.dumps({"invalid": 3}), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_packet(packet)
