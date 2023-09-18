#!/usr/bin/env python3
import argparse
import os
import sys
import json
import random
import string
import datetime
from prettytable import PrettyTable
import nltk

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from docsbot.base import Base
from docsbot.config import CONFIG

if hasattr(CONFIG.env, "HTTP_PROXY_FOR_GLOBAL_ACCESS"):
    nltk.set_proxy(CONFIG.env.HTTP_PROXY_FOR_GLOBAL_ACCESS)
elif hasattr(CONFIG.env, "OPENAI_PROXY"):
    nltk.set_proxy(CONFIG.env.OPENAI_PROXY)


class ChatBase:
    def __init__(self):
        self.bases_file = CONFIG.bases_file
        if os.path.exists(self.bases_file):
            with open(self.bases_file, 'r') as f:
                self.base = json.load(f)
        else:
            self.base = {}

    def save_base(self):
        with open(self.bases_file, 'w') as f:
            json.dump(self.base, f)

    def addbase(self, path):
        if os.path.isdir(path):
            base_id = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
            base_id = f"base000{base_id.lower()}"
            self.base[base_id] = {'location': path}
            base = Base(base_id)
            docs = base.add(path)
            if not docs:
                print("No valid documents found in the directory")
                return
            self.base[base_id]['file_count'] = len(docs)
            self.base[base_id]['files'] = docs
            self.base[base_id]['vector_store_type'] = base.vector_store_type
            # created is the time the base was created
            self.base[base_id]['created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_base()
            print(f"Successfully added {len(docs)} document(s) in '{path}' to '{base_id}'!")
        else:
            print("Invalid directory")

    def listbase(self):
        table = PrettyTable()
        table.field_names = ["ID", "Location", "Count", "Store", "Created"]
        for _id, base in self.base.items():
            table.add_row([_id, base['location'],
                           "N/A" if 'file_count' not in base else f"{base['file_count']} files",
                           "N/A" if 'vector_store_type' not in base else base['vector_store_type'],
                           "N/A" if 'created' not in base else base['created']
                           ])
        print(table)

    def deletebase(self, base_ids):
        for base_id in base_ids:
            if base_id in self.base:
                base = Base(base_id, self.base[base_id]['vector_store_type'])
                base.delete()
                del self.base[base_id]
                self.save_base()
                print(f"Successfully deleted '{base_id}'")
            else:
                print(f"Invalid base ID {base_id}")


    def _pretty_print_query_result(self, data):
        if data['source_documents']:
            print(f"【答】：{data['answer']}")
            print("【来源】：")
        else:
            print("没有找到相关文件")
            return
        # 根据来源整理文档内容
        source_dict = {}
        for doc in data['source_documents']:
            content = doc.page_content.replace('\n', ' ')
            if not doc.metadata:
                source = "Unknown"
            else:
                source = doc.metadata['source']
            if source in source_dict:
                source_dict[source].append(content)
            else:
                source_dict[source] = [content]

        # 打印整理后的文档内容
        for i, (source, contents) in enumerate(source_dict.items()):
            print(f"{i + 1}. 文件：{source}")
            for j, content in enumerate(contents, start=1):
                print(f"   内容{j}. {content}")


    def check(self):
        from qdrant_client import QdrantClient
        client = QdrantClient(url='http://127.0.0.1:6333')
        print(client.get_collections())
        print(os.environ)

    def query(self, base_id, debug=False):
        if debug:
            import langchain
            langchain.debug = True
        if base_id not in self.base:
            print("Invalid base ID")
            return
        base = Base(base_id, self.base[base_id]['vector_store_type'])
        # 根据用户的交互式输入，调用多次query方法
        while True:
            question = input("请输入您的问题：")
            if question == 'exit':
                break
            self._pretty_print_query_result(base.query(question))

def main():
    chat_base = ChatBase()

    parser = argparse.ArgumentParser(prog='chatbase')
    subparsers = parser.add_subparsers(dest='command')

    with open(os.path.join(os.path.dirname(__file__), "version")) as f:
        app_version = f.read()
    # Add version argument
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {app_version}')

    parser_addbase = subparsers.add_parser('addbase')
    parser_addbase.add_argument('path', type=str)

    parser_listbase = subparsers.add_parser('listbase')

    parser_deletebase = subparsers.add_parser('deletebase')
    parser_deletebase.add_argument('base_ids', nargs='+')

    # parser_query = subparsers.add_parser('query')
    # parser_query.add_argument('base_id', type=str)
    # parser_query.add_argument('query', type=str)

    parser_showconfig = subparsers.add_parser('showconfig')

    parser_query = subparsers.add_parser('query')
    parser_query.add_argument('base_id', type=str)
    parser_query.add_argument('--debug', action='store_true')
    # parser_query.add_argument('--thread_id', type=str)


    args = parser.parse_args()

    if args.command == 'addbase':
        chat_base.addbase(args.path)
    elif args.command == 'listbase':
        chat_base.listbase()
    elif args.command == 'deletebase':
        chat_base.deletebase(args.base_ids)
    elif args.command == 'showconfig':
        CONFIG.print()
    elif args.command == 'query':
        chat_base.query(args.base_id,
                        # thread_id=args.thread_id,
                        debug=args.debug
                        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
