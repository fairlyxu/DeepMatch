"""
# File       : test_user_file.py
# Time       ：2024/10/30 20:29
# Author     ：fairyxu 
# Description：
"""
import unittest
import pandas as pd


# 假设 get_var_feature 是您的函数
def get_var_feature(values, embedding_name, max_len):
    # 这是一个示例函数，您可以替换为实际的实现
    return [embedding_name + str(value) for value in values[:max_len]]


class TestDefaultColumnValue(unittest.TestCase):
    def setUp(self):
        # 创建一个示例 DataFrame
        self.data = pd.DataFrame({
            'existing_column': [1, 2, 3],
            'another_column': [4, 5, 6]
        })
        self.embedding_name = 'embed_'
        self.max_len = 2
        self.default_value = 'default_value_here'
        self.f_name = 'missing_column'

    def test_add_default_column(self):
        # 检查列是否存在，如果不存在则添加默认值
        if self.f_name not in self.data.columns:
            self.data[self.f_name] = self.default_value

        # 确保列名已添加，并填充了默认值
        self.assertIn(self.f_name, self.data.columns)
        self.assertTrue((self.data[self.f_name] == self.default_value).all())

        # 调用 get_var_feature 函数进行测试
        var_feature_list = get_var_feature(self.data[self.f_name].values, self.embedding_name, self.max_len)

        # 验证返回值
        expected_result = [self.embedding_name + self.default_value] * self.max_len
        self.assertEqual(var_feature_list, expected_result)


if __name__ == '__main__':
    unittest.main()
