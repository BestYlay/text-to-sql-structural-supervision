import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from text2sql.schema import ast_labels
from text2sql.checks import inspect_sql
from text2sql.feedback import diagnostics,fresh_messages
from text2sql.runtime import metadata_formats,accepted
from text2sql.metrics import result_sets_match,extract_sql
from text2sql.sqlite import execute_readonly

ROOT=Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        structural,checking=metadata_formats(json.loads((ROOT/'examples/schema.json').read_text()))
        self.schema=structural['toy'];self.meta=checking['toy']

    def test_ast_labels_for_bridge_join(self):
        labels=ast_labels('SELECT c.name FROM client c JOIN disp d ON c.client_id=d.client_id JOIN account a ON a.account_id=d.account_id',self.schema)
        self.assertEqual(labels['tables'],['account','client','disp'])
        self.assertEqual(len(labels['edges']),2)
        self.assertIn(('client','name'),labels['columns'])

    def test_alias_scope_cte_and_missing_field(self):
        labels=ast_labels('WITH x AS (SELECT name FROM client) SELECT x.name FROM x',self.schema)
        self.assertIn(('client','name'),labels['columns'])
        with self.assertRaises(Exception):ast_labels('SELECT account.name FROM account',self.schema)

    def test_join_warning_and_schema_only_prompt(self):
        sql='SELECT c.name FROM client c JOIN account a ON c.client_id=a.account_id'
        issues=inspect_sql(sql,self.meta)
        self.assertIn('join_path',[i['code'] for i in issues])
        feedback=diagnostics(sql,self.meta,issues)
        self.assertIn('disp',feedback)
        messages=fresh_messages([dict(role='system',content='system'),dict(role='user',content='schema question')],feedback)
        self.assertEqual([m['role'] for m in messages],['system','user'])
        self.assertNotIn(sql,messages[1]['content'])

    def test_acceptance_does_not_use_reference(self):
        pred=dict(completed=True,generation_status='ok',pred_sql='SELECT 1')
        self.assertTrue(accepted(pred,[]))
        self.assertFalse(accepted(dict(pred,generation_status='constraint_dead_end'),[]))
        self.assertFalse(accepted(pred,[dict(severity='error')]))
        self.assertTrue(accepted(pred,[dict(severity='warning')]))

    def test_author_comparison_is_string_bag(self):
        self.assertTrue(result_sets_match([(1,)],[('1',)]))
        self.assertFalse(result_sets_match([(1,),(1,)],[(1,)]))
        self.assertEqual(extract_sql('```sql\nSELECT 1;\n```'),'SELECT 1')

    def test_readonly_and_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'toy.sqlite'
            con=sqlite3.connect(db);con.execute('CREATE TABLE t(x INTEGER)');con.execute('INSERT INTO t VALUES (1)');con.commit();con.close()
            self.assertEqual(execute_readonly(db,'SELECT x FROM t')['rows'],[(1,)])
            self.assertEqual(execute_readonly(db,'DELETE FROM t')['status'],'error')
            self.assertEqual(execute_readonly(db,'SELECT 1; SELECT 2')['status'],'error')
            query='WITH RECURSIVE x(v) AS (SELECT 1 UNION ALL SELECT v+1 FROM x) SELECT sum(v) FROM x'
            self.assertEqual(execute_readonly(db,query,timeout=.2)['status'],'timeout')


if __name__=='__main__':unittest.main()
