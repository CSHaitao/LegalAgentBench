## 运行
```python
pip install -r requirments.txt
cd src
python react.py
```

- data/dataset.json: 数据集
- data/law_dataset.xlsx: 数据表格（信息源）
- src/schcma.py: 表格字段说明
- src/api.py: 所有工具实现
- src/generated_tools: 包装api成tools，供大模型使用
- src/react.py: 程序入口，react实现