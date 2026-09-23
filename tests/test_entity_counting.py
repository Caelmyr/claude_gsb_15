import sys
import tempfile
import types
import unittest
from unittest import mock

# 测试环境可能没有安装 jieba；被测的计数逻辑不依赖实际分词结果。
jieba = types.ModuleType('jieba')
jieba.cut = lambda text: list(text)
jieba.analyse = types.SimpleNamespace(extract_tags=lambda text, topK=10: [])
sys.modules.setdefault('jieba', jieba)
sys.modules.setdefault('jieba.analyse', jieba.analyse)
posseg = types.ModuleType('jieba.posseg')
posseg.cut = lambda text: []
sys.modules.setdefault('jieba.posseg', posseg)

from backend.nlp.ner import NamedEntityRecognizer
from backend.nlp.pipeline import NLPPipeline
from backend.graph import storage as storage_module
from backend.graph.storage import GraphStorage
from backend.graph import builder as builder_module
from backend.graph.builder import GraphBuilder
from backend import utils


class EntityCountingTest(unittest.TestCase):
    def test_ner_preserves_repeated_occurrences(self):
        text = '北京很好。北京很好。北京很好。北京很好。北京很好。'

        entities = NamedEntityRecognizer().recognize(text)

        self.assertEqual([entity['text'] for entity in entities], ['北京'] * 5)
        self.assertTrue(all(entity['type'] == 'LOCATION' for entity in entities))

    def test_nested_pattern_does_not_count_same_span_twice(self):
        entities = NamedEntityRecognizer().recognize('北京是一个大城市。')

        self.assertEqual([entity['text'] for entity in entities], ['北京'])

    def test_pipeline_reports_actual_occurrence_count(self):
        text = '北京是一个大城市。北京是一个大城市。北京是一个大城市。北京是一个大城市。北京是一个大城市。'

        result = NLPPipeline().process(text)

        beijing = [entity for entity in result['entities'] if entity['text'] == '北京']
        self.assertEqual(len(beijing), 1)
        self.assertEqual(beijing[0]['count'], 5)

    def test_replacing_same_document_does_not_accumulate_counts(self):
        entities = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 5,
            'context': '北京'
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', entities, [])
                self.assertEqual(storage.get_entity('北京')['count'], 5)

                for _ in range(9):
                    storage.replace_document_data('doc1', entities, [])

                self.assertEqual(storage.get_entity('北京')['count'], 5)
                self.assertEqual(storage.get_entity('北京')['doc_counts'], {'doc1': 5})

    def test_counts_are_recalculated_across_documents(self):
        first_doc = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 5,
            'context': '北京'
        }]
        second_doc = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 2,
            'context': '北京'
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', first_doc, [])
                storage.replace_document_data('doc2', second_doc, [])
                self.assertEqual(storage.get_entity('北京')['count'], 7)

                first_doc[0]['count'] = 3
                storage.replace_document_data('doc1', first_doc, [])
                self.assertEqual(storage.get_entity('北京')['count'], 5)

    def test_relations_do_not_inflate_entity_counts(self):
        entities = [
            {'text': '北京', 'type': 'LOCATION', 'count': 5},
            {'text': '上海', 'type': 'LOCATION', 'count': 1},
        ]
        relations = [{
            'subject': '北京',
            'subject_type': 'LOCATION',
            'predicate': '相关',
            'object': '上海',
            'object_type': 'LOCATION',
            'properties': {'source_text': '北京和上海相关'}
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', entities, relations)

                self.assertEqual(storage.get_entity('北京')['count'], 5)
                self.assertEqual(storage.get_entity('上海')['count'], 1)

                storage.add_relation('北京', 'LOCATION', '位于', '上海', 'LOCATION')
                self.assertEqual(storage.get_entity('北京')['count'], 5)
                self.assertEqual(storage.get_entity('上海')['count'], 1)

    def test_manual_entity_count_is_retained_during_document_reparse(self):
        entities = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 5
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', entities, [])
                storage.add_entity('北京', 'LOCATION', {'manual': True})
                self.assertEqual(storage.get_entity('北京')['count'], 6)

                storage.replace_document_data('doc1', entities, [])
                self.assertEqual(storage.get_entity('北京')['count'], 6)
                self.assertEqual(storage.get_entity('北京')['manual_count'], 1)

    def test_idempotent_counts_are_persisted_on_storage_restart(self):
        entities = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 5
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', entities, [])

                reloaded_storage = GraphStorage()
                self.assertEqual(reloaded_storage.get_entity('北京')['count'], 5)
                self.assertEqual(
                    reloaded_storage.get_entity('北京')['doc_counts'],
                    {'doc1': 5}
                )

    def test_relation_uses_existing_entity_shard_type(self):
        entities = [{
            'text': '北京',
            'type': 'LOCATION',
            'count': 5
        }]

        with tempfile.TemporaryDirectory() as graph_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir):
                storage = GraphStorage()
                storage.replace_document_data('doc1', entities, [])
                storage.add_relation('北京', 'OTHER', '相关', '上海', 'LOCATION')

                self.assertEqual(storage.get_entity('北京')['type'], 'LOCATION')
                self.assertIn('北京', storage._cache['LOCATION']['entities'])
                self.assertNotIn('北京', storage._cache['OTHER']['entities'])


class FakePipeline:
    def __init__(self, result):
        self.result = result

    def process(self, text):
        return self.result


class GraphBuilderCountingTest(unittest.TestCase):
    def test_builder_reparse_is_idempotent(self):
        result = {
            'entities': [{
                'text': '北京',
                'type': 'LOCATION',
                'count': 5,
                'context': '北京'
            }],
            'relations': [],
            'triples': []
        }

        with tempfile.TemporaryDirectory() as graph_dir, \
                tempfile.TemporaryDirectory() as triples_dir:
            with mock.patch.object(utils.config, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(storage_module, 'GRAPH_DIR', graph_dir), \
                    mock.patch.object(utils.config, 'TRIPLES_DIR', triples_dir), \
                    mock.patch.object(builder_module, 'TRIPLES_DIR', triples_dir), \
                    mock.patch.object(builder_module, 'DOCUMENTS_DIR', triples_dir):
                storage = GraphStorage()
                builder = GraphBuilder(storage)
                builder.nlp_pipeline = FakePipeline(result)

                builder.build_from_text('北京出现五次的文档', 'doc1')
                builder.build_from_text('北京出现五次的文档', 'doc1')

                self.assertEqual(storage.get_entity('北京')['count'], 5)


if __name__ == '__main__':
    unittest.main()
