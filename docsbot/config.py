import os
from dotenv import load_dotenv, set_key, dotenv_values
from prettytable import PrettyTable


class Env:
    def __init__(self):
        pass


class Config:
    def __init__(self):
        self.home_dir = os.path.join(os.path.expanduser('~'), '.docsbot')
        self.config_file = os.path.join(self.home_dir, 'docsbot.env')
        self.bases_file = os.path.join(self.home_dir, 'base_data.json')
        self.vectors_dir = os.path.join(self.home_dir, 'vectors')
        self.env = Env()

        if not os.path.exists(self.home_dir):
            os.makedirs(self.home_dir)
        if not os.path.exists(self.vectors_dir):
            os.makedirs(self.vectors_dir)

        load_dotenv(self.config_file)

        for key, value in os.environ.items():
            setattr(self.env, key, value)

        if hasattr(self.env, 'OPENAI_API_KEY'):
            openai_key = self.env.OPENAI_API_KEY
        else:
            openai_key = input('请输入您的 OpenAI Key: ')
            setattr(self.env, 'OPENAI_API_KEY', openai_key)

        set_key(self.config_file, 'OPENAI_API_KEY', openai_key)
        load_dotenv(self.config_file)


    def print(self):
        table1 = PrettyTable(align='l')
        table1.field_names = ["配置项", "值"]
        table1.add_row(["Home", self.home_dir])
        table1.add_row(["Config file", self.config_file])
        table1.add_row(["Base meta file", self.bases_file])
        table1.add_row(["Vector store dir", self.vectors_dir])

        table2 = PrettyTable(align='l', title="配置文件")
        table2.field_names = ["配置项", "值"]
        for _k, _v in dotenv_values(self.config_file).items():
            table2.add_row([_k, _v])
        print(table1)
        print(table2)


CONFIG = Config()



