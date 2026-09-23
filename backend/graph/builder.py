"""
图谱构建模块 - 从文档构建知识图谱
"""
import json
import os
from typing import List, Dict
from backend.nlp.pipeline import NLPPipeline
from backend.graph.storage import GraphStorage
from backend.utils.config import TRIPLES_DIR, DOCUMENTS_DIR


class GraphBuilder:
    """图谱构建器"""

    def __init__(self, storage: GraphStorage = None):
        self.nlp_pipeline = NLPPipeline()
        self.storage = storage or GraphStorage()
        self._rebuild_saved_documents()

    def _rebuild_saved_documents(self):
        """从已保存的文档/解析结果恢复各文档贡献，兼容旧版本累计计数数据。"""
        if not os.path.isdir(TRIPLES_DIR):
            return

        for filename in os.listdir(TRIPLES_DIR):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(TRIPLES_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                doc_id = cached.get('doc_id')
                if not doc_id:
                    continue

                doc_info_path = os.path.join(DOCUMENTS_DIR, f'{doc_id}.json')
                if os.path.exists(doc_info_path):
                    with open(doc_info_path, 'r', encoding='utf-8') as f:
                        doc_info = json.load(f)
                    source_path = os.path.join(DOCUMENTS_DIR, doc_info['stored_filename'])
                    if os.path.exists(source_path):
                        from backend.utils.text_extractor import extract_text
                        result = self.nlp_pipeline.process(extract_text(source_path))
                        entities = result['entities']
                        relations = result['relations']
                    else:
                        entities = cached.get('entities', [])
                        relations = cached.get('relations', [])
                else:
                    entities = cached.get('entities', [])
                    relations = cached.get('relations', [])

                if entities or relations:
                    self.storage.replace_document_data(doc_id, entities, relations)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                # 损坏的解析缓存不应阻止系统启动；用户可重新解析该文档。
                continue

    def build_from_text(self, text: str, doc_id: str = None) -> Dict:
        """从文本构建图谱"""
        # NLP处理
        result = self.nlp_pipeline.process(text)

        if doc_id:
            # 按文档整体替换，重复解析同一文档时不会累加旧计数。
            self.storage.replace_document_data(
                doc_id,
                result['entities'],
                result['relations']
            )
        else:
            # 兼容直接解析临时文本的调用；有文档ID的解析走上面的幂等路径。
            for entity in result['entities']:
                self.storage.add_entity(
                    entity['text'],
                    entity['type'],
                    {'context': entity.get('context', '')},
                    count=entity.get('count', 1)
                )

            # 添加关系到图谱；关系只引用实体，不重复增加实体出现次数。
            for relation in result['relations']:
                self.storage.add_relation(
                    relation['subject'],
                    relation['subject_type'],
                    relation['predicate'],
                    relation['object'],
                    relation['object_type'],
                    {'source_text': relation.get('source_text', '')}
                )

        return {
            'doc_id': doc_id,
            'entities_count': len(result['entities']),
            'relations_count': len(result['relations']),
            'triples': result['triples'],
            'entities': result['entities'],
            'relations': result['relations']
        }

    def build_from_document(self, doc_path: str, doc_id: str) -> Dict:
        """从文档文件构建图谱"""
        from backend.utils.text_extractor import extract_text
        text = extract_text(doc_path)
        return self.build_from_text(text, doc_id)

    def add_triple(self, subject: str, subject_type: str, predicate: str,
                   obj: str, object_type: str) -> Dict:
        """手动添加三元组"""
        self.storage.add_relation(subject, subject_type, predicate, obj, object_type)
        return {'status': 'success', 'triple': (subject, predicate, obj)}

    def get_graph_data(self) -> Dict:
        """获取图谱可视化数据"""
        return self.storage.get_graph_data()

    def get_statistics(self) -> Dict:
        """获取图谱统计"""
        return self.storage.get_statistics()

    def query(self, query_text: str) -> Dict:
        """查询图谱"""
        from backend.graph.query import GraphQuery
        query_engine = GraphQuery(self.storage)
        return query_engine.query_keyword(query_text)
