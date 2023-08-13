# DocsBot 使用说明

DocsBot 是一个简单的命令行小工具，用对话的方式，快速查询本地的文档库（Word、PDF、PPT等）

## 快速开始 Quick Start

### 安装方法1：通过pip 安装
```shell
$ pip install docsbot -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装方法2：通过Docker运行
```shell
docker run -ti  -e "OPENAI_PROXY=http://192.168.3.112:8001" -e "OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx" jingstory/docsbot  
```

### 使用
```
$ docsbot
Please enter your OpenAI Key: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
usage: chatbase [-h] {addbase,listbase,deletebase,query} ...

positional arguments:
  {addbase,listbase,deletebase,query}

options:
  -h, --help            show this help message and exit
  
  
$ docsbot addbase /Users/jeffrey/Downloads/laws

Using vector store:  Qdrant
loading from dir: /Users/jeffrey/Downloads/laws
Added 4 document(s) to base base000x7uvrvegk9vv
Added /Users/jeffrey/Downloads/laws with ID base000x7uvrvegk9vv

$ docsbot listbase
+---------------------+-------------------------------+---------+
|          ID         |            Location           |  Count  |
+---------------------+-------------------------------+---------+
| base000x7uvrvegk9vv | /Users/jeffrey/Downloads/laws | 4 files |
+---------------------+-------------------------------+---------+

$ docsbot query base000x7uvrvegk9vv

Using vector store:  [Qdrant]-base000sxgohnz1b2r8@http://127.0.0.1:6333 
请输入您的问题：我租了3年，没给钱，合法吗
【回答】： 不合法。根据第721条，承租人应当按照约定的期限支付租金，如果没有约定或者约定不明确，租赁期限不满一年的，应当在租赁期限届满时支付；租赁期限一年以上的，应当在每届满一年时支付，剩余期限不满一年的，应当在租赁期限届满时支付。因此，没有按照约定的期限支付租金是不合法的。
【来源】：
1. 文件：/tmp/laws/中华人民共和国民法典.docx
   内容1. 第七百二十一条  承租人应当按照约定的期限支付租金。对支付租金的期限没有约定或者约定不明确，依据本法第五百一十条的规定仍不能确定，租赁期限不满一年的，应当在租赁期限届满时支付；租赁期限一年以上的，应当在每届满一年时支付，剩余期限不满一年的，应当在租赁期限届满时支付。
   内容2. 第七百二十二条  承租人无正当理由未支付或者迟延支付租金的，出租人可以请求承租人在合理期限内支付；承租人逾期不支付的，出租人可以解除合同。  第七百二十三条  因第三人主张权利，致使承租人不能对租赁物使用、收益的，承租人可以请求减少租金或者不支付租金。  第三人主张权利的，承租人应当及时通知出租人。  

```


## 命令和参数

以下是 DocsBot 支持的命令及其参数：

```bash
$ docsbot addbase <dir>          # 用于添加一个新的资料库。 `<dir>`: 要添加的资料库的目录路径。
$ docsbot listbase               # 用于列出所有已添加的资料库。
$ docsbot deletebase <baseid>    # 用于删除一个已添加的资料库。 `<baseid>`: 要删除的资料库的ID。
$ docsbot query <baseid> [--debug] # 用于查询一个资料库。 `<baseid>`: 要查询的资料库的ID。 `--debug`: 是否显示调试信息。
$ docsbot showconfig             # 用于显示当前的配置信息。
```



## 配置项

### Home目录
`docsbot` 默认使用目录 `$HOME/.docsbot`来存储自己的配置信息、资料库的元信息与索引数据等。
```python
# 该目录中的文件和文件夹
docsbot.env  -- 配置文件
base_data.json -- 资料库的元信息，比如ID、目录、文件数
vectors -- 对资料库Embedding后的向量索引数据的存储目录

```

### 配置文件
第一次运行时，请根据提示设置OpenAI的Key，`docsbot`自动保存到配置文件 
`$HOME/.docsbot/docsbot.env`中。

所有的配置项如下：
```env
OPENAI_API_KEY=xxxxxxxxx   # OpenAI的API Key，必须（第一次运行时，会提示输入）
OPENAI_PROXY="http://192.168.3.112:8001" # 代理，可选
HTTP_PROXY_FOR_GLOBAL_ACCESS="http://192.168.3.112:8001"  # 默认使用OPENAI_PROXY
VECTOR_STORE_TYPE="Chroma"  # 索引类型，目前支持Chroma、Qdrant
QDRANT_SERVER_URL="http://192.168.1.22:6333" # Qdrant的地址，当VECTOR_STORE_TYPE="Qdrant"时必须
```


## FAQ

### Q：由于众所周知的原因，如何设置OPENAI的代理？
在配置文件中，增加如下配置项：
```env
OPENAI_PROXY="http://192.168.3.112:8001"
```

### Q：如何更换向量数据库为Qdrant？

在配置文件中，增加如下配置项（修改为真实的Qdrant地址）：
```env
VECTOR_STORE_TYPE="Qdrant"
QDRANT_SERVER_URL="http://192.168.1.22:6333"
```
### Q：如何打开调试模式？

在命令行中增加`--debug`参数即可。

```shell
$ docsbot query base000x7uvrvegk9vv --debug
```


# Release Notes

详见：[文档](releasenotes.md)