"""
图谱存储模块 - JSON文件分片存储

实体出现次数（count）按来源记账，避免重复解析文档时计数累加：
- doc_counts:  {doc_id: 该文档内的出现次数}，重新解析时整体替换该文档的贡献
- manual_count: 手动标注/接口直接添加产生的次数
- count = sum(doc_counts.values()) + manual_count

关系通过 doc_ids 记录来源文档：重新解析时只移除该文档独有的关系，
被其他文档或手动标注共享的关系会保留。
"""
import json
import os
import threading
from typing import Dict, List, Optional
from backend.utils.config import GRAPH_DIR, GRAPH_SHARDS


class GraphStorage:
    """图谱存储管理器 - 按实体类型分片"""

    def __init__(self):
        self.lock = threading.RLock()
        self._ensure_directories()
        self._cache = {}
        self._load_all_shards()

    def _ensure_directories(self):
        """确保目录存在"""
        os.makedirs(GRAPH_DIR, exist_ok=True)

    def _load_all_shards(self):
        """加载所有分片到缓存，并兼容旧格式数据"""
        for entity_type, filename in GRAPH_SHARDS.items():
            filepath = os.path.join(GRAPH_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    shard = json.load(f)
            else:
                shard = {'entities': {}, 'relations': []}

            shard.setdefault('entities', {})
            shard.setdefault('relations', [])

            for entity in shard['entities'].values():
                entity.setdefault('properties', {})
                entity.setdefault('doc_counts', {})
                entity.setdefault('manual_count', 0)
                # 兼容旧数据：旧格式只有累计的 count、没有来源明细。
                # 仅在完全缺失来源明细时，把历史计数保留为 manual_count
                # （不归属任何文档，重新解析不会清除它）
                legacy_count = entity.pop('count', 0)
                if legacy_count and not entity['doc_counts'] \
                        and not entity['manual_count']:
                    entity['manual_count'] = legacy_count
                self._recompute_count(entity)

            for relation in shard['relations']:
                relation.setdefault('properties', {})
                relation.setdefault('doc_ids', [])

            self._cache[entity_type] = shard

    def _save_shard(self, entity_type: str):
        """保存指定分片到文件（调用方需持有锁）"""
        if entity_type not in self._cache:
            return
        filename = GRAPH_SHARDS.get(entity_type, 'other.json')
        filepath = os.path.join(GRAPH_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._cache[entity_type], f, ensure_ascii=False, indent=2)

    @staticmethod
    def _recompute_count(entity: Dict):
        """根据各来源贡献重新计算实体出现次数"""
        entity['count'] = sum(entity.get('doc_counts', {}).values()) \
            + entity.get('manual_count', 0)

    def _ensure_entity(self, entity_text: str, entity_type: str) -> Dict:
        """确保实体存在并返回其数据，但不改变出现次数（供关系添加使用）"""
        shard = self._cache.setdefault(
            entity_type, {'entities': {}, 'relations': []}
        )
        if entity_text not in shard['entities']:
            shard['entities'][entity_text] = {
                'id': self._next_entity_id(entity_type),
                'text': entity_text,
                'type': entity_type,
                'properties': {},
                'doc_counts': {},
                'manual_count': 0,
                'count': 0
            }
        return shard['entities'][entity_text]

    def _next_entity_id(self, entity_type: str) -> str:
        """生成分片内不与现有ID冲突的实体ID"""
        existing = {
            e['id'] for e in self._cache.get(entity_type, {})
            .get('entities', {}).values()
        }
        index = len(existing)
        new_id = f"{entity_type}_{index}"
        while new_id in existing:
            index += 1
            new_id = f"{entity_type}_{index}"
        return new_id

    def add_entity(self, entity_text: str, entity_type: str,
                   properties: Dict = None, count: int = 1):
        """添加实体（手动标注 / 无文档来源的增量添加）

        计入 manual_count，不会被后续文档重新解析清除。
        """
        with self.lock:
            entity = self._ensure_entity(entity_text, entity_type)
            entity['manual_count'] = entity.get('manual_count', 0) + max(1, count)
            if properties:
                entity['properties'].update(properties)
            self._recompute_count(entity)
            self._save_shard(entity_type)

    def merge_document(self, doc_id: str, entities: List[Dict],
                       relations: List[Dict]):
        """用某文档的最新解析结果整体替换该文档对图谱的贡献

        重新解析文档时先撤销该文档的旧贡献（实体出现次数、来源关系、
        无引用的空实体），再写入新结果，计数因此不会逐次累加。
        """
        touched = set()
        with self.lock:
            self._remove_document_contribution(doc_id, touched)

            # 写入新的实体出现次数
            for entity in entities:
                text = entity['text']
                entity_type = entity['type']
                occurrence = max(1, entity.get('count', 1))
                touched.add(entity_type)

                record = self._ensure_entity(text, entity_type)
                record['doc_counts'][doc_id] = occurrence
                record['properties'].update({
                    'context': entity.get('context', ''),
                    'doc_id': doc_id
                })
                self._recompute_count(record)

            # 写入新的关系（记录来源文档，重复关系合并 doc_ids）
            for rel in relations:
                subject_type = rel['subject_type']
                object_type = rel['object_type']
                touched.add(subject_type)

                self._ensure_entity(rel['subject'], subject_type)
                self._ensure_entity(rel['object'], object_type)

                shard = self._cache[subject_type]
                existing = self._find_relation(
                    shard, rel['subject'], rel['predicate'], rel['object']
                )
                if existing is None:
                    shard['relations'].append({
                        'subject': rel['subject'],
                        'subject_type': subject_type,
                        'predicate': rel['predicate'],
                        'object': rel['object'],
                        'object_type': object_type,
                        'properties': {
                            'source_text': rel.get('source_text', ''),
                            'doc_id': doc_id
                        },
                        'doc_ids': [doc_id]
                    })
                elif doc_id not in existing['doc_ids']:
                    existing['doc_ids'].append(doc_id)

            for entity_type in touched:
                self._save_shard(entity_type)

    def _remove_document_contribution(self, doc_id: str, touched: set):
        """撤销某文档对实体计数和关系的旧贡献（调用方需持有锁）"""
        # 移除该文档独有的关系
        for shard in self._cache.values():
            for relation in list(shard['relations']):
                if doc_id in relation.get('doc_ids', []):
                    touched.add(relation['subject_type'])
                    touched.add(relation['object_type'])
                    relation['doc_ids'].remove(doc_id)
                    relation['doc_ids'].sort()
                    # 没有任何来源（其他文档或手动标注）的关系才删除
                    if not relation['doc_ids'] and not relation.get('properties', {}).get('manual'):
                        shard['relations'].remove(relation)

        # 撤销实体出现次数，并回收无贡献且无关系引用的实体
        for entity_type, shard in self._cache.items():
            for text in list(shard['entities'].keys()):
                entity = shard['entities'][text]
                if doc_id in entity.get('doc_counts', {}):
                    del entity['doc_counts'][doc_id]
                    self._recompute_count(entity)
                    touched.add(entity_type)

                if entity['count'] <= 0 and not self._entity_is_referenced(text):
                    touched.add(entity_type)
                    del shard['entities'][text]

    @staticmethod
    def _find_relation(shard: Dict, subject: str, predicate: str,
                       obj: str) -> Optional[Dict]:
        """在分片内查找指定三元组关系"""
        for relation in shard['relations']:
            if (relation['subject'] == subject
                    and relation['predicate'] == predicate
                    and relation['object'] == obj):
                return relation
        return None

    def _entity_is_referenced(self, entity_text: str) -> bool:
        """实体是否还被任意关系引用"""
        for shard in self._cache.values():
            for relation in shard['relations']:
                if relation['subject'] == entity_text or relation['object'] == entity_text:
                    return True
        return False

    def add_relation(self, subject: str, subject_type: str, predicate: str,
                     obj: str, object_type: str, properties: Dict = None):
        """手动添加关系

        关系表示实体间的关联，不是实体的一次"出现"，因此不再增加实体计数，
        避免一个实体因参与多条关系而被虚增出现次数。
        """
        with self.lock:
            # 确保实体存在（不改变出现次数）
            self._ensure_entity(subject, subject_type)
            self._ensure_entity(obj, object_type)

            shard = self._cache[subject_type]
            existing = self._find_relation(shard, subject, predicate, obj)
            if existing is None:
                relation = {
                    'subject': subject,
                    'subject_type': subject_type,
                    'predicate': predicate,
                    'object': obj,
                    'object_type': object_type,
                    'properties': properties or {},
                    'doc_ids': []
                }
                shard['relations'].append(relation)
                self._save_shard(subject_type)
                self._save_shard(object_type)
            elif properties:
                existing['properties'].update(properties)
                self._save_shard(subject_type)

    def get_entity(self, entity_text: str) -> Optional[Dict]:
        """获取实体信息"""
        with self.lock:
            for shard in self._cache.values():
                if entity_text in shard['entities']:
                    return shard['entities'][entity_text]
        return None

    def get_entity_relations(self, entity_text: str) -> List[Dict]:
        """获取实体的所有关系"""
        relations = []
        with self.lock:
            for shard in self._cache.values():
                for relation in shard['relations']:
                    if relation['subject'] == entity_text or relation['object'] == entity_text:
                        relations.append(relation)
        return relations

    def get_all_entities(self) -> List[Dict]:
        """获取所有实体"""
        entities = []
        with self.lock:
            for shard in self._cache.values():
                entities.extend(shard['entities'].values())
        return entities

    def get_all_relations(self) -> List[Dict]:
        """获取所有关系"""
        relations = []
        with self.lock:
            for shard in self._cache.values():
                relations.extend(shard['relations'])
        return relations

    def get_graph_data(self) -> Dict:
        """获取图谱可视化数据"""
        nodes = []
        links = []
        node_ids = set()

        with self.lock:
            for entity_type, shard in self._cache.items():
                for entity_text, entity_data in shard['entities'].items():
                    if entity_data['id'] not in node_ids:
                        node_ids.add(entity_data['id'])
                        nodes.append({
                            'id': entity_data['id'],
                            'label': entity_text,
                            'type': entity_type,
                            'count': entity_data.get('count', 0)
                        })

                for relation in shard['relations']:
                    source_entity = self.get_entity(relation['subject'])
                    target_entity = self.get_entity(relation['object'])
                    if source_entity and target_entity:
                        links.append({
                            'source': source_entity['id'],
                            'target': target_entity['id'],
                            'label': relation['predicate']
                        })

        return {'nodes': nodes, 'links': links}

    def search_entities(self, keyword: str) -> List[Dict]:
        """搜索实体"""
        results = []
        with self.lock:
            for shard in self._cache.values():
                for entity_text, entity_data in shard['entities'].items():
                    if keyword in entity_text:
                        results.append(entity_data)
        return results

    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        total_entities = 0
        total_relations = 0
        entity_counts = {}

        with self.lock:
            for entity_type, shard in self._cache.items():
                count = len(shard['entities'])
                entity_counts[entity_type] = count
                total_entities += count
                total_relations += len(shard['relations'])

        return {
            'total_entities': total_entities,
            'total_relations': total_relations,
            'entity_counts': entity_counts
        }
