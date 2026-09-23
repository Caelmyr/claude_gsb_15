"""
图谱存储模块 - JSON文件分片存储
"""
import json
import os
import threading
from typing import Dict, List, Optional
from backend.utils.config import GRAPH_DIR, GRAPH_SHARDS

ADHOC_SOURCE = '__adhoc__'


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

    @staticmethod
    def _normalize_entity_type(entity_type: str) -> str:
        """将未配置分片的实体类型归入 OTHER 分片。"""
        return entity_type if entity_type in GRAPH_SHARDS else 'OTHER'

    @staticmethod
    def _normalize_entity(entity: Dict) -> Dict:
        """兼容旧数据，并把计数拆成可按文档重算的结构。"""
        normalized = dict(entity)
        properties = dict(normalized.get('properties') or {})
        normalized['properties'] = properties

        raw_doc_counts = normalized.get('doc_counts')
        if isinstance(raw_doc_counts, dict):
            doc_counts = {
                str(doc_id): int(count)
                for doc_id, count in raw_doc_counts.items()
                if int(count) > 0
            }
        else:
            doc_counts = {}

        manual_count = int(normalized.get('manual_count', 0) or 0)
        if not raw_doc_counts and properties.get('manual'):
            # 旧版本的人工标注没有 manual_count，原 count 即人工标注次数。
            manual_count = max(0, int(normalized.get('count', manual_count) or 0))

        normalized['doc_counts'] = doc_counts
        normalized['manual_count'] = max(0, manual_count)
        normalized['count'] = sum(doc_counts.values()) + normalized['manual_count']
        return normalized

    @staticmethod
    def _normalize_relation(relation: Dict) -> Dict:
        """兼容旧关系数据，补充文档来源列表。"""
        normalized = dict(relation)
        properties = dict(normalized.get('properties') or {})
        normalized['properties'] = properties

        raw_doc_ids = normalized.get('doc_ids')
        if isinstance(raw_doc_ids, list):
            doc_ids = [str(doc_id) for doc_id in raw_doc_ids if doc_id]
        else:
            doc_id = properties.get('doc_id')
            doc_ids = [str(doc_id)] if doc_id else []
        normalized['doc_ids'] = list(dict.fromkeys(doc_ids))
        return normalized

    def _load_all_shards(self):
        """加载所有分片到缓存"""
        for entity_type, filename in GRAPH_SHARDS.items():
            filepath = os.path.join(GRAPH_DIR, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    shard = json.load(f)
            else:
                shard = {}

            entities = shard.get('entities', {})
            for entity_text, entity in entities.items():
                entities[entity_text] = self._normalize_entity(entity)

            relations = [
                self._normalize_relation(relation)
                for relation in shard.get('relations', [])
            ]
            self._cache[entity_type] = {
                'entities': entities,
                'relations': relations
            }

    def _save_shard(self, entity_type: str):
        """保存指定分片到文件"""
        entity_type = self._normalize_entity_type(entity_type)
        if entity_type not in self._cache:
            self._cache[entity_type] = {'entities': {}, 'relations': []}

        filename = GRAPH_SHARDS.get(entity_type, 'other.json')
        filepath = os.path.join(GRAPH_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._cache[entity_type], f, ensure_ascii=False, indent=2)

    def _get_shard(self, entity_type: str) -> Dict:
        entity_type = self._normalize_entity_type(entity_type)
        if entity_type not in self._cache:
            self._cache[entity_type] = {'entities': {}, 'relations': []}
        return self._cache[entity_type]

    @staticmethod
    def _recalculate_entity_count(entity: Dict):
        """根据各文档贡献和人工标注贡献重新计算总出现次数。"""
        entity['count'] = sum(entity.get('doc_counts', {}).values()) + entity.get('manual_count', 0)

    def _next_entity_id(self, shard: Dict, entity_type: str) -> str:
        """生成分片内唯一的实体ID。"""
        existing_ids = {
            entity.get('id')
            for entity in shard['entities'].values()
            if entity.get('id')
        }
        index = len(shard['entities'])
        entity_id = f"{entity_type}_{index}"
        while entity_id in existing_ids:
            index += 1
            entity_id = f"{entity_type}_{index}"
        return entity_id

    def _find_entity_locked(self, entity_text: str):
        for shard_entity_type, shard in self._cache.items():
            if entity_text in shard['entities']:
                return shard_entity_type, shard['entities'][entity_text]
        return None, None

    def _ensure_entity_locked(self, entity_text: str, entity_type: str,
                              properties: Dict = None) -> Dict:
        """确保关系引用的实体存在；关系本身不计入实体出现次数。"""
        entity_type = self._normalize_entity_type(entity_type)
        existing_type, entity = self._find_entity_locked(entity_text)
        if entity is not None:
            if properties:
                entity['properties'].update(properties)
            return entity

        shard = self._get_shard(entity_type)
        entity = {
            'id': self._next_entity_id(shard, entity_type),
            'text': entity_text,
            'type': entity_type,
            'properties': properties or {},
            'doc_counts': {},
            'manual_count': 0,
            'count': 0
        }
        shard['entities'][entity_text] = entity
        return entity

    def add_entity(self, entity_text: str, entity_type: str,
                   properties: Dict = None, count: int = 1):
        """
        添加实体。

        人工标注（properties['manual'] 为 True）计入 manual_count；没有文档ID的
        普通写入计入临时来源。文档解析应使用 replace_document_data，以保证重复
        解析同一文档时按最新结果重算，而不是累加。
        """
        properties = dict(properties or {})
        count = max(0, int(count or 0))

        with self.lock:
            entity_type = self._normalize_entity_type(entity_type)
            shard = self._get_shard(entity_type)
            normalized_type, existing = self._find_entity_locked(entity_text)
            if existing is None:
                entity = {
                    'id': self._next_entity_id(shard, entity_type),
                    'text': entity_text,
                    'type': entity_type,
                    'properties': properties,
                    'doc_counts': {},
                    'manual_count': 0,
                    'count': 0
                }
                shard['entities'][entity_text] = entity
            else:
                entity_type = normalized_type
                shard = self._get_shard(entity_type)
                entity = existing
                entity['properties'].update(properties)

            if count:
                if properties.get('manual'):
                    entity['manual_count'] = entity.get('manual_count', 0) + count
                else:
                    source = properties.get('doc_id') or ADHOC_SOURCE
                    entity['doc_counts'][source] = entity.get('doc_counts', {}).get(source, 0) + count
            self._recalculate_entity_count(entity)
            self._save_shard(entity_type)

    def _find_relation_locked(self, subject: str, predicate: str, obj: str):
        for entity_type, shard in self._cache.items():
            for index, relation in enumerate(shard['relations']):
                if (relation['subject'] == subject and
                        relation['predicate'] == predicate and
                        relation['object'] == obj):
                    return entity_type, index, relation
        return None, None, None

    def _add_relation_locked(self, subject: str, subject_type: str, predicate: str,
                             obj: str, object_type: str, properties: Dict = None,
                             doc_id: Optional[str] = None,
                             replace_properties: bool = False):
        normalized_subject_type, subject_entity = self._find_entity_locked(subject)
        if subject_entity is not None:
            subject_type = normalized_subject_type
        else:
            subject_type = self._normalize_entity_type(subject_type)

        normalized_object_type, object_entity = self._find_entity_locked(obj)
        if object_entity is not None:
            object_type = normalized_object_type
        else:
            object_type = self._normalize_entity_type(object_type)

        self._ensure_entity_locked(subject, subject_type)
        self._ensure_entity_locked(obj, object_type)

        shard_name, _, existing = self._find_relation_locked(subject, predicate, obj)
        if existing:
            if replace_properties:
                existing['properties'] = dict(properties or {})
            else:
                existing['properties'].update(properties or {})
            if doc_id and doc_id not in existing['doc_ids']:
                existing['doc_ids'].append(doc_id)
            return existing

        relation = {
            'subject': subject,
            'subject_type': subject_type,
            'predicate': predicate,
            'object': obj,
            'object_type': object_type,
            'properties': properties or {},
            'doc_ids': [doc_id] if doc_id else []
        }
        self._get_shard(subject_type)['relations'].append(relation)
        return relation

    def add_relation(self, subject: str, subject_type: str, predicate: str,
                     obj: str, object_type: str, properties: Dict = None):
        """添加人工关系；关系不会额外增加实体出现次数。"""
        with self.lock:
            relation = self._add_relation_locked(
                subject, subject_type, predicate, obj, object_type, properties
            )
            self._save_shard(relation['subject_type'])

    def replace_document_data(self, doc_id: str, entities: List[Dict],
                              relations: List[Dict]):
        """
        用一次文档解析结果替换该文档的旧贡献。

        同一文档重复解析时，先移除旧的实体次数和关系，再写入本次结果，
        保证出现次数与当前文档内容一致。
        """
        doc_id = str(doc_id)
        affected_shards = set()

        with self.lock:
            # 移除该文档的旧关系。关系按全局三元组去重，并记录来源文档。
            for shard_name, shard in self._cache.items():
                remaining_relations = []
                for relation in shard['relations']:
                    if doc_id in relation.get('doc_ids', []):
                        relation['doc_ids'] = [
                            source for source in relation['doc_ids'] if source != doc_id
                        ]
                        affected_shards.add(shard_name)
                        if relation['doc_ids']:
                            remaining_relations.append(relation)
                    else:
                        remaining_relations.append(relation)
                shard['relations'] = remaining_relations

            # 移除实体中该文档的旧出现次数。
            for shard_name, shard in self._cache.items():
                for entity in shard['entities'].values():
                    if doc_id in entity.get('doc_counts', {}):
                        entity['doc_counts'].pop(doc_id, None)
                        self._recalculate_entity_count(entity)
                        affected_shards.add(shard_name)

            # 写入本次解析得到的实体出现次数。
            for entity_data in entities:
                entity_text = entity_data['text']
                entity_type = self._normalize_entity_type(entity_data['type'])
                occurrence_count = int(entity_data.get('count', 1) or 1)
                if occurrence_count <= 0:
                    continue

                properties = dict(entity_data.get('properties') or {})
                properties.update({'doc_id': doc_id})
                if entity_data.get('context'):
                    properties['context'] = entity_data['context']

                existing_type, entity = self._find_entity_locked(entity_text)
                if entity is None:
                    shard = self._get_shard(entity_type)
                    entity = {
                        'id': self._next_entity_id(shard, entity_type),
                        'text': entity_text,
                        'type': entity_type,
                        'properties': {},
                        'doc_counts': {},
                        'manual_count': 0,
                        'count': 0
                    }
                    shard['entities'][entity_text] = entity
                else:
                    entity_type = existing_type
                    shard = self._get_shard(entity_type)

                entity['properties'].update(properties)
                entity['doc_counts'][doc_id] = occurrence_count
                self._recalculate_entity_count(entity)
                affected_shards.add(entity_type)

            # 写入本次解析得到的关系；同名三元组跨文档只保留一条并记录来源。
            for relation_data in relations:
                properties = dict(relation_data.get('properties') or {})
                properties['doc_id'] = doc_id
                relation = self._add_relation_locked(
                    relation_data['subject'],
                    relation_data['subject_type'],
                    relation_data['predicate'],
                    relation_data['object'],
                    relation_data['object_type'],
                    properties,
                    doc_id,
                    replace_properties=True
                )
                affected_shards.add(relation['subject_type'])

            self._cleanup_unreferenced_entities()

            # 关系删除可能使其他分片中的宾语实体失去引用，保存所有分片以清理这些节点。
            affected_shards.update(self._cache.keys())
            for shard_name in affected_shards:
                self._save_shard(shard_name)

    def _cleanup_unreferenced_entities(self, affected_shards: set = None):
        """删除没有任何出现次数且不再被关系引用的实体。"""
        referenced = set()
        for shard in self._cache.values():
            for relation in shard['relations']:
                referenced.add((relation['subject'], relation['subject_type']))
                referenced.add((relation['object'], relation['object_type']))

        shard_names = affected_shards if affected_shards else set(self._cache.keys())
        for shard_name in shard_names:
            shard = self._get_shard(shard_name)
            shard['entities'] = {
                entity_text: entity
                for entity_text, entity in shard['entities'].items()
                if entity.get('count', 0) > 0
                or (entity_text, shard_name) in referenced
            }

    def get_entity(self, entity_text: str) -> Optional[Dict]:
        """获取实体信息"""
        with self.lock:
            for entity_type, shard in self._cache.items():
                if entity_text in shard['entities']:
                    return shard['entities'][entity_text]
        return None

    def get_entity_relations(self, entity_text: str) -> List[Dict]:
        """获取实体的所有关系"""
        with self.lock:
            relations = []
            for entity_type, shard in self._cache.items():
                for relation in shard['relations']:
                    if relation['subject'] == entity_text or relation['object'] == entity_text:
                        relations.append(relation)
            return relations

    def get_all_entities(self) -> List[Dict]:
        """获取所有实体"""
        with self.lock:
            entities = []
            for entity_type, shard in self._cache.items():
                entities.extend(shard['entities'].values())
            return entities

    def get_all_relations(self) -> List[Dict]:
        """获取所有关系"""
        with self.lock:
            relations = []
            for entity_type, shard in self._cache.items():
                relations.extend(shard['relations'])
            return relations

    def get_graph_data(self) -> Dict:
        """获取图谱可视化数据"""
        with self.lock:
            nodes = []
            links = []
            node_ids = set()

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
        with self.lock:
            results = []
            for entity_type, shard in self._cache.items():
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
