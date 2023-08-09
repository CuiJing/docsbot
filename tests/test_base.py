import os
import sys

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_DIR)
EXAMPLE_DIR = os.path.join(PROJECT_DIR, "example_docs")  # For testing purposes

import unittest
from docsbot.base import Base



class TestHighLevelFunctions(unittest.TestCase):
    # 假设您有一些示例数据的路径
    sample_data_path = EXAMPLE_DIR

    def setUp(self):
        # 首先删除可能存在的Base，确保测试环境的清洁
        self.base_qdrant = Base("test_base_qdrant", "Qdrant")
        self.base_chroma = Base("test_base_chroma", "Chroma")
        self.base_qdrant.delete()
        self.base_chroma.delete()

    def test_high_level_functions(self):
        # 1. 向Base添加数据
        self.base_qdrant.add(self.sample_data_path)
        self.base_chroma.add(self.sample_data_path)

        # # 2. 查询数据
        # response_qdrant = self.base_qdrant.query("什么是技术开发合同")
        # response_chroma = self.base_chroma.query("什么是技术开发合同")

        # # 判断基于您期望的返回类型/内容进行验证
        # self.assertIsNotNone(response_qdrant)
        # self.assertIsNotNone(response_chroma)

    def tearDown(self):
        # 5. 删除Base
        self.base_qdrant.delete()
        self.base_chroma.delete()


if __name__ == '__main__':
    unittest.main()
