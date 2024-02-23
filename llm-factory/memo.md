
## パッケージのインストール

```bash
# パッケージのインストール
!git clone https://github.com/hiyouga/LLaMA-Factory.git
%cd LLaMA-Factory
!pip install -r requirements.txt
!pip install bitsandbytes
```

```bash
Cloning into 'LLaMA-Factory'...
remote: Enumerating objects: 6979, done.
remote: Counting objects: 100% (998/998), done.
remote: Compressing objects: 100% (361/361), done.
remote: Total 6979 (delta 686), reused 903 (delta 634), pack-reused 5981
Receiving objects: 100% (6979/6979), 204.86 MiB | 28.88 MiB/s, done.
Resolving deltas: 100% (5037/5037), done.
Updating files: 100% (140/140), done.
/content/LLaMA-Factory
Requirement already satisfied: torch>=1.13.1 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 1)) (2.1.0+cu121)
Collecting transformers>=4.37.2 (from -r requirements.txt (line 2))
  Downloading transformers-4.37.2-py3-none-any.whl (8.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.4/8.4 MB 28.0 MB/s eta 0:00:00
Collecting datasets>=2.14.3 (from -r requirements.txt (line 3))
  Downloading datasets-2.17.0-py3-none-any.whl (536 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.6/536.6 kB 39.4 MB/s eta 0:00:00
Collecting accelerate>=0.21.0 (from -r requirements.txt (line 4))
  Downloading accelerate-0.26.1-py3-none-any.whl (270 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 270.9/270.9 kB 37.7 MB/s eta 0:00:00
Collecting peft>=0.7.0 (from -r requirements.txt (line 5))
  Downloading peft-0.8.2-py3-none-any.whl (183 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 183.4/183.4 kB 27.0 MB/s eta 0:00:00
Collecting trl>=0.7.6 (from -r requirements.txt (line 6))
  Downloading trl-0.7.10-py3-none-any.whl (150 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 150.9/150.9 kB 13.7 MB/s eta 0:00:00
Collecting gradio<4.0.0,>=3.38.0 (from -r requirements.txt (line 7))
  Downloading gradio-3.50.2-py3-none-any.whl (20.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.3/20.3 MB 73.2 MB/s eta 0:00:00
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 8)) (1.11.4)
Collecting einops (from -r requirements.txt (line 9))
  Downloading einops-0.7.0-py3-none-any.whl (44 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 44.6/44.6 kB 7.3 MB/s eta 0:00:00
Requirement already satisfied: sentencepiece in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 10)) (0.1.99)
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 11)) (3.20.3)
Requirement already satisfied: jieba in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 12)) (0.42.1)
Collecting rouge-chinese (from -r requirements.txt (line 13))
  Downloading rouge_chinese-1.0.3-py3-none-any.whl (21 kB)
Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 14)) (3.8.1)
Collecting uvicorn (from -r requirements.txt (line 15))
  Downloading uvicorn-0.27.0.post1-py3-none-any.whl (60 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.7/60.7 kB 9.5 MB/s eta 0:00:00
Requirement already satisfied: pydantic in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 16)) (2.6.1)
Collecting fastapi (from -r requirements.txt (line 17))
  Downloading fastapi-0.109.2-py3-none-any.whl (92 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 92.1/92.1 kB 14.6 MB/s eta 0:00:00
Collecting sse-starlette (from -r requirements.txt (line 18))
  Downloading sse_starlette-2.0.0-py3-none-any.whl (9.0 kB)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 19)) (3.7.1)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (3.13.1)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (4.9.0)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (1.12)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (3.2.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (3.1.3)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (2023.6.0)
Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.13.1->-r requirements.txt (line 1)) (2.1.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.19.3 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (0.20.3)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (1.23.5)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (23.2)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (6.0.1)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (2023.12.25)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (2.31.0)
Requirement already satisfied: tokenizers<0.19,>=0.14 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (0.15.1)
Requirement already satisfied: safetensors>=0.4.1 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (0.4.2)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers>=4.37.2->-r requirements.txt (line 2)) (4.66.1)
Collecting pyarrow>=12.0.0 (from datasets>=2.14.3->-r requirements.txt (line 3))
  Downloading pyarrow-15.0.0-cp310-cp310-manylinux_2_28_x86_64.whl (38.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.3/38.3 MB 44.3 MB/s eta 0:00:00
Requirement already satisfied: pyarrow-hotfix in /usr/local/lib/python3.10/dist-packages (from datasets>=2.14.3->-r requirements.txt (line 3)) (0.6)
Collecting dill<0.3.9,>=0.3.0 (from datasets>=2.14.3->-r requirements.txt (line 3))
  Downloading dill-0.3.8-py3-none-any.whl (116 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 116.3/116.3 kB 17.3 MB/s eta 0:00:00
Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets>=2.14.3->-r requirements.txt (line 3)) (1.5.3)
Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets>=2.14.3->-r requirements.txt (line 3)) (3.4.1)
Collecting multiprocess (from datasets>=2.14.3->-r requirements.txt (line 3))
  Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 134.8/134.8 kB 20.8 MB/s eta 0:00:00
Requirement already satisfied: aiohttp in /usr/local/lib/python3.10/dist-packages (from datasets>=2.14.3->-r requirements.txt (line 3)) (3.9.3)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate>=0.21.0->-r requirements.txt (line 4)) (5.9.5)
Collecting tyro>=0.5.11 (from trl>=0.7.6->-r requirements.txt (line 6))
  Downloading tyro-0.7.2-py3-none-any.whl (79 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 79.8/79.8 kB 12.8 MB/s eta 0:00:00
Collecting aiofiles<24.0,>=22.0 (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading aiofiles-23.2.1-py3-none-any.whl (15 kB)
Requirement already satisfied: altair<6.0,>=4.2.0 in /usr/local/lib/python3.10/dist-packages (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (4.2.2)
Collecting ffmpy (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading ffmpy-0.3.1.tar.gz (5.5 kB)
  Preparing metadata (setup.py) ... done
Collecting gradio-client==0.6.1 (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading gradio_client-0.6.1-py3-none-any.whl (299 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 299.2/299.2 kB 39.5 MB/s eta 0:00:00
Collecting httpx (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading httpx-0.26.0-py3-none-any.whl (75 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.9/75.9 kB 13.2 MB/s eta 0:00:00
Requirement already satisfied: importlib-resources<7.0,>=1.3 in /usr/local/lib/python3.10/dist-packages (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (6.1.1)
Requirement already satisfied: markupsafe~=2.0 in /usr/local/lib/python3.10/dist-packages (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (2.1.5)
Collecting orjson~=3.0 (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading orjson-3.9.13-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (138 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 138.7/138.7 kB 21.0 MB/s eta 0:00:00
Requirement already satisfied: pillow<11.0,>=8.0 in /usr/local/lib/python3.10/dist-packages (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (9.4.0)
Collecting pydub (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading pydub-0.25.1-py2.py3-none-any.whl (32 kB)
Collecting python-multipart (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading python_multipart-0.0.7-py3-none-any.whl (22 kB)
Collecting semantic-version~=2.0 (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
Collecting websockets<12.0,>=10.0 (from gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading websockets-11.0.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (129 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.9/129.9 kB 18.2 MB/s eta 0:00:00
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from rouge-chinese->-r requirements.txt (line 13)) (1.16.0)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk->-r requirements.txt (line 14)) (8.1.7)
Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk->-r requirements.txt (line 14)) (1.3.2)
Collecting h11>=0.8 (from uvicorn->-r requirements.txt (line 15))
  Downloading h11-0.14.0-py3-none-any.whl (58 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 9.0 MB/s eta 0:00:00
Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic->-r requirements.txt (line 16)) (0.6.0)
Requirement already satisfied: pydantic-core==2.16.2 in /usr/local/lib/python3.10/dist-packages (from pydantic->-r requirements.txt (line 16)) (2.16.2)
Collecting starlette<0.37.0,>=0.36.3 (from fastapi->-r requirements.txt (line 17))
  Downloading starlette-0.36.3-py3-none-any.whl (71 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.5/71.5 kB 12.3 MB/s eta 0:00:00
Requirement already satisfied: anyio in /usr/local/lib/python3.10/dist-packages (from sse-starlette->-r requirements.txt (line 18)) (3.7.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (4.48.1)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (1.4.5)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (3.1.1)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->-r requirements.txt (line 19)) (2.8.2)
Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (0.4)
Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (4.19.2)
Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (0.12.1)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (23.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (1.4.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (6.0.5)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (1.9.4)
Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp->datasets>=2.14.3->-r requirements.txt (line 3)) (4.0.3)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets>=2.14.3->-r requirements.txt (line 3)) (2023.4)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.37.2->-r requirements.txt (line 2)) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.37.2->-r requirements.txt (line 2)) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.37.2->-r requirements.txt (line 2)) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers>=4.37.2->-r requirements.txt (line 2)) (2024.2.2)
Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio->sse-starlette->-r requirements.txt (line 18)) (1.3.0)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio->sse-starlette->-r requirements.txt (line 18)) (1.2.0)
Collecting docstring-parser>=0.14.1 (from tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6))
  Downloading docstring_parser-0.15-py3-none-any.whl (36 kB)
Requirement already satisfied: rich>=11.1.0 in /usr/local/lib/python3.10/dist-packages (from tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6)) (13.7.0)
Collecting shtab>=1.5.6 (from tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6))
  Downloading shtab-1.6.5-py3-none-any.whl (13 kB)
Collecting httpcore==1.* (from httpx->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7))
  Downloading httpcore-1.0.2-py3-none-any.whl (76 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 76.9/76.9 kB 12.6 MB/s eta 0:00:00
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.13.1->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (2023.12.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (0.33.0)
Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6.0,>=4.2.0->gradio<4.0.0,>=3.38.0->-r requirements.txt (line 7)) (0.17.1)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1.0->tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6)) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=11.1.0->tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6)) (2.16.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=11.1.0->tyro>=0.5.11->trl>=0.7.6->-r requirements.txt (line 6)) (0.1.2)
Building wheels for collected packages: ffmpy
  Building wheel for ffmpy (setup.py) ... done
  Created wheel for ffmpy: filename=ffmpy-0.3.1-py3-none-any.whl size=5579 sha256=83623eccaeb846686259e2cfea5d3e826634c490f6fe09bbcd8f222faa23d622
  Stored in directory: /root/.cache/pip/wheels/01/a6/d1/1c0828c304a4283b2c1639a09ad86f83d7c487ef34c6b4a1bf
Successfully built ffmpy
Installing collected packages: pydub, ffmpy, websockets, shtab, semantic-version, rouge-chinese, python-multipart, pyarrow, orjson, h11, einops, docstring-parser, dill, aiofiles, uvicorn, starlette, multiprocess, httpcore, tyro, sse-starlette, httpx, fastapi, accelerate, transformers, gradio-client, datasets, trl, peft, gradio
  Attempting uninstall: pyarrow
    Found existing installation: pyarrow 10.0.1
    Uninstalling pyarrow-10.0.1:
      Successfully uninstalled pyarrow-10.0.1
  Attempting uninstall: transformers
    Found existing installation: transformers 4.35.2
    Uninstalling transformers-4.35.2:
      Successfully uninstalled transformers-4.35.2
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
lida 0.0.10 requires kaleido, which is not installed.
ibis-framework 7.1.0 requires pyarrow<15,>=2, but you have pyarrow 15.0.0 which is incompatible.
Successfully installed accelerate-0.26.1 aiofiles-23.2.1 datasets-2.17.0 dill-0.3.8 docstring-parser-0.15 einops-0.7.0 fastapi-0.109.2 ffmpy-0.3.1 gradio-3.50.2 gradio-client-0.6.1 h11-0.14.0 httpcore-1.0.2 httpx-0.26.0 multiprocess-0.70.16 orjson-3.9.13 peft-0.8.2 pyarrow-15.0.0 pydub-0.25.1 python-multipart-0.0.7 rouge-chinese-1.0.3 semantic-version-2.10.0 shtab-1.6.5 sse-starlette-2.0.0 starlette-0.36.3 transformers-4.37.2 trl-0.7.10 tyro-0.7.2 uvicorn-0.27.0.post1 websockets-11.0.3
Collecting bitsandbytes
  Downloading bitsandbytes-0.42.0-py3-none-any.whl (105.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.0/105.0 MB 15.9 MB/s eta 0:00:00
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from bitsandbytes) (1.11.4)
Requirement already satisfied: numpy<1.28.0,>=1.21.6 in /usr/local/lib/python3.10/dist-packages (from scipy->bitsandbytes) (1.23.5)
Installing collected packages: bitsandbytes
Successfully installed bitsandbytes-0.42.0
```


## LLaMA-Factoryの起動

```bash
# LLaMA-Factoryの起動
!CUDA_VISIBLE_DEVICES=0 python src/train_web.py
```

```bash
2024-02-09 14:32:13.552214: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-02-09 14:32:13.552257: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-02-09 14:32:13.553985: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-02-09 14:32:14.777903: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://1880c749478f2359be.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
02/09/2024 14:36:25 - WARNING - llmtuner.hparams.parser - We recommend enable `upcast_layernorm` in quantized training.
02/09/2024 14:36:25 - WARNING - llmtuner.hparams.parser - `ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.
[INFO|training_args.py:1828] 2024-02-09 14:36:25,143 >> PyTorch: setting up devices
/usr/local/lib/python3.10/dist-packages/transformers/training_args.py:1741: FutureWarning: `--push_to_hub_token` is deprecated and will be removed in version 5 of 🤗 Transformers. Use `--hub_token` instead.
  warnings.warn(
02/09/2024 14:36:25 - INFO - llmtuner.hparams.parser - Process rank: 0, device: cuda:0, n_gpu: 1
  distributed training: True, compute dtype: torch.bfloat16
02/09/2024 14:36:25 - INFO - llmtuner.hparams.parser - Training/evaluation parameters Seq2SeqTrainingArguments(
_n_gpu=1,
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
auto_find_batch_size=False,
bf16=True,
bf16_full_eval=False,
data_seed=None,
dataloader_drop_last=False,
dataloader_num_workers=0,
dataloader_persistent_workers=False,
dataloader_pin_memory=True,
ddp_backend=None,
ddp_broadcast_buffers=None,
ddp_bucket_cap_mb=None,
ddp_find_unused_parameters=False,
ddp_timeout=1800,
debug=[],
deepspeed=None,
disable_tqdm=True,
dispatch_batches=None,
do_eval=False,
do_predict=False,
do_train=True,
eval_accumulation_steps=None,
eval_delay=0,
eval_steps=None,
evaluation_strategy=no,
fp16=False,
fp16_backend=auto,
fp16_full_eval=False,
fp16_opt_level=O1,
fsdp=[],
fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
fsdp_min_num_params=0,
fsdp_transformer_layer_cls_to_wrap=None,
full_determinism=False,
generation_config=None,
generation_max_length=None,
generation_num_beams=None,
gradient_accumulation_steps=4,
gradient_checkpointing=False,
gradient_checkpointing_kwargs=None,
greater_is_better=None,
group_by_length=False,
half_precision_backend=auto,
hub_always_push=False,
hub_model_id=None,
hub_private_repo=False,
hub_strategy=every_save,
hub_token=<HUB_TOKEN>,
ignore_data_skip=False,
include_inputs_for_metrics=False,
include_num_input_tokens_seen=False,
include_tokens_per_second=False,
jit_mode_eval=False,
label_names=None,
label_smoothing_factor=0.0,
learning_rate=0.0002,
length_column_name=length,
load_best_model_at_end=False,
local_rank=0,
log_level=passive,
log_level_replica=warning,
log_on_each_node=True,
logging_dir=saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/runs/Feb09_14-36-25_51a765157995,
logging_first_step=False,
logging_nan_inf_filter=True,
logging_steps=5,
logging_strategy=steps,
lr_scheduler_kwargs={},
lr_scheduler_type=cosine,
max_grad_norm=0.3,
max_steps=-1,
metric_for_best_model=None,
mp_parameters=,
neftune_noise_alpha=None,
no_cuda=False,
num_train_epochs=3.0,
optim=adamw_torch,
optim_args=None,
output_dir=saves/LLaMA-7B/lora/train_2024-02-09-14-34-22,
overwrite_output_dir=False,
past_index=-1,
per_device_eval_batch_size=8,
per_device_train_batch_size=16,
predict_with_generate=False,
prediction_loss_only=False,
push_to_hub=False,
push_to_hub_model_id=None,
push_to_hub_organization=None,
push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
ray_scope=last,
remove_unused_columns=True,
report_to=['tensorboard'],
resume_from_checkpoint=None,
run_name=saves/LLaMA-7B/lora/train_2024-02-09-14-34-22,
save_on_each_node=False,
save_only_model=False,
save_safetensors=True,
save_steps=100,
save_strategy=steps,
save_total_limit=None,
seed=42,
skip_memory_metrics=True,
sortish_sampler=False,
split_batches=False,
tf32=None,
torch_compile=False,
torch_compile_backend=None,
torch_compile_mode=None,
torchdynamo=None,
tpu_metrics_debug=False,
tpu_num_cores=None,
use_cpu=False,
use_ipex=False,
use_legacy_prediction_loop=False,
use_mps_device=False,
warmup_ratio=0.0,
warmup_steps=0,
weight_decay=0.0,
)
tokenizer_config.json: 100% 725/725 [00:00<00:00, 4.51MB/s]
tokenizer.model: 100% 500k/500k [00:00<00:00, 15.9MB/s]
special_tokens_map.json: 100% 437/437 [00:00<00:00, 3.30MB/s]
tokenizer.json: 100% 1.84M/1.84M [00:00<00:00, 27.3MB/s]
[INFO|tokenization_utils_base.py:2027] 2024-02-09 14:36:26,290 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.model
[INFO|tokenization_utils_base.py:2027] 2024-02-09 14:36:26,290 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2027] 2024-02-09 14:36:26,290 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/special_tokens_map.json
[INFO|tokenization_utils_base.py:2027] 2024-02-09 14:36:26,290 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer_config.json
[INFO|tokenization_utils_base.py:2027] 2024-02-09 14:36:26,290 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.json
config.json: 100% 641/641 [00:00<00:00, 4.78MB/s]
[INFO|configuration_utils.py:729] 2024-02-09 14:36:26,614 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 14:36:26,615 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

02/09/2024 14:36:26 - INFO - llmtuner.model.patcher - Quantizing model to 4 bit.
pytorch_model.bin.index.json: 100% 26.8k/26.8k [00:00<00:00, 112MB/s]
[INFO|modeling_utils.py:3476] 2024-02-09 14:36:27,247 >> loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/pytorch_model.bin.index.json
Downloading shards:   0% 0/2 [00:00<?, ?it/s]
pytorch_model-00001-of-00002.bin:   0% 0.00/9.98G [00:00<?, ?B/s]
pytorch_model-00001-of-00002.bin:   0% 10.5M/9.98G [00:00<02:47, 59.6MB/s]
pytorch_model-00001-of-00002.bin:   0% 21.0M/9.98G [00:00<02:12, 75.4MB/s]
pytorch_model-00001-of-00002.bin:   0% 41.9M/9.98G [00:00<01:37, 102MB/s] 
pytorch_model-00001-of-00002.bin:   1% 62.9M/9.98G [00:00<01:21, 122MB/s]
pytorch_model-00001-of-00002.bin:   1% 83.9M/9.98G [00:00<01:09, 142MB/s]
pytorch_model-00001-of-00002.bin:   1% 105M/9.98G [00:00<01:04, 153MB/s] 
pytorch_model-00001-of-00002.bin:   1% 126M/9.98G [00:00<01:00, 164MB/s]
pytorch_model-00001-of-00002.bin:   1% 147M/9.98G [00:01<00:57, 170MB/s]
pytorch_model-00001-of-00002.bin:   2% 168M/9.98G [00:01<00:55, 175MB/s]
pytorch_model-00001-of-00002.bin:   2% 189M/9.98G [00:01<00:54, 178MB/s]
pytorch_model-00001-of-00002.bin:   2% 210M/9.98G [00:01<00:54, 180MB/s]
pytorch_model-00001-of-00002.bin:   2% 231M/9.98G [00:01<00:53, 182MB/s]
pytorch_model-00001-of-00002.bin:   3% 252M/9.98G [00:01<00:52, 184MB/s]
pytorch_model-00001-of-00002.bin:   3% 273M/9.98G [00:01<00:52, 184MB/s]
pytorch_model-00001-of-00002.bin:   3% 294M/9.98G [00:01<00:53, 181MB/s]
pytorch_model-00001-of-00002.bin:   3% 315M/9.98G [00:01<00:52, 183MB/s]
pytorch_model-00001-of-00002.bin:   3% 336M/9.98G [00:02<00:52, 183MB/s]
pytorch_model-00001-of-00002.bin:   4% 357M/9.98G [00:02<00:52, 184MB/s]
pytorch_model-00001-of-00002.bin:   4% 377M/9.98G [00:02<00:52, 184MB/s]
pytorch_model-00001-of-00002.bin:   4% 398M/9.98G [00:02<00:52, 181MB/s]
pytorch_model-00001-of-00002.bin:   4% 419M/9.98G [00:02<00:52, 181MB/s]
pytorch_model-00001-of-00002.bin:   4% 440M/9.98G [00:02<00:52, 183MB/s]
pytorch_model-00001-of-00002.bin:   5% 461M/9.98G [00:02<00:52, 181MB/s]
pytorch_model-00001-of-00002.bin:   5% 482M/9.98G [00:02<00:52, 181MB/s]
pytorch_model-00001-of-00002.bin:   5% 503M/9.98G [00:02<00:51, 183MB/s]
pytorch_model-00001-of-00002.bin:   5% 524M/9.98G [00:03<00:51, 184MB/s]
pytorch_model-00001-of-00002.bin:   5% 545M/9.98G [00:03<00:55, 170MB/s]
pytorch_model-00001-of-00002.bin:   6% 566M/9.98G [00:03<00:54, 174MB/s]
pytorch_model-00001-of-00002.bin:   6% 587M/9.98G [00:03<00:52, 178MB/s]
pytorch_model-00001-of-00002.bin:   6% 608M/9.98G [00:03<00:52, 178MB/s]
pytorch_model-00001-of-00002.bin:   6% 629M/9.98G [00:03<00:51, 181MB/s]
pytorch_model-00001-of-00002.bin:   7% 650M/9.98G [00:03<00:51, 182MB/s]
pytorch_model-00001-of-00002.bin:   7% 671M/9.98G [00:03<00:51, 181MB/s]
pytorch_model-00001-of-00002.bin:   7% 692M/9.98G [00:04<00:51, 181MB/s]
pytorch_model-00001-of-00002.bin:   7% 713M/9.98G [00:04<00:50, 182MB/s]
pytorch_model-00001-of-00002.bin:   7% 734M/9.98G [00:04<00:50, 183MB/s]
pytorch_model-00001-of-00002.bin:   8% 755M/9.98G [00:04<00:50, 184MB/s]
pytorch_model-00001-of-00002.bin:   8% 776M/9.98G [00:04<00:51, 179MB/s]
pytorch_model-00001-of-00002.bin:   8% 797M/9.98G [00:04<00:50, 183MB/s]
pytorch_model-00001-of-00002.bin:   8% 818M/9.98G [00:04<00:49, 183MB/s]
pytorch_model-00001-of-00002.bin:   8% 839M/9.98G [00:04<00:49, 184MB/s]
pytorch_model-00001-of-00002.bin:   9% 860M/9.98G [00:04<00:50, 181MB/s]
pytorch_model-00001-of-00002.bin:   9% 881M/9.98G [00:05<00:49, 183MB/s]
pytorch_model-00001-of-00002.bin:   9% 902M/9.98G [00:05<00:50, 181MB/s]
pytorch_model-00001-of-00002.bin:   9% 923M/9.98G [00:05<00:49, 183MB/s]
pytorch_model-00001-of-00002.bin:   9% 944M/9.98G [00:05<00:48, 185MB/s]
pytorch_model-00001-of-00002.bin:  10% 965M/9.98G [00:05<00:48, 185MB/s]
pytorch_model-00001-of-00002.bin:  10% 986M/9.98G [00:05<00:48, 184MB/s]
pytorch_model-00001-of-00002.bin:  10% 1.01G/9.98G [00:05<00:48, 185MB/s]
pytorch_model-00001-of-00002.bin:  10% 1.03G/9.98G [00:05<00:48, 185MB/s]
pytorch_model-00001-of-00002.bin:  11% 1.05G/9.98G [00:05<00:49, 182MB/s]
pytorch_model-00001-of-00002.bin:  11% 1.07G/9.98G [00:06<00:48, 183MB/s]
pytorch_model-00001-of-00002.bin:  11% 1.09G/9.98G [00:06<00:48, 183MB/s]
pytorch_model-00001-of-00002.bin:  11% 1.11G/9.98G [00:06<00:48, 184MB/s]
pytorch_model-00001-of-00002.bin:  11% 1.13G/9.98G [00:06<00:48, 182MB/s]
pytorch_model-00001-of-00002.bin:  12% 1.15G/9.98G [00:06<00:48, 180MB/s]
pytorch_model-00001-of-00002.bin:  12% 1.17G/9.98G [00:06<00:47, 184MB/s]
pytorch_model-00001-of-00002.bin:  12% 1.20G/9.98G [00:06<00:47, 184MB/s]
pytorch_model-00001-of-00002.bin:  12% 1.22G/9.98G [00:06<00:47, 185MB/s]
pytorch_model-00001-of-00002.bin:  12% 1.24G/9.98G [00:07<00:47, 185MB/s]
pytorch_model-00001-of-00002.bin:  13% 1.26G/9.98G [00:07<00:47, 185MB/s]
pytorch_model-00001-of-00002.bin:  13% 1.28G/9.98G [00:07<00:46, 185MB/s]
pytorch_model-00001-of-00002.bin:  13% 1.30G/9.98G [00:07<00:46, 185MB/s]
pytorch_model-00001-of-00002.bin:  13% 1.32G/9.98G [00:07<00:46, 184MB/s]
pytorch_model-00001-of-00002.bin:  13% 1.34G/9.98G [00:07<00:46, 184MB/s]
pytorch_model-00001-of-00002.bin:  14% 1.36G/9.98G [00:07<00:46, 184MB/s]
pytorch_model-00001-of-00002.bin:  14% 1.38G/9.98G [00:07<00:46, 184MB/s]
pytorch_model-00001-of-00002.bin:  14% 1.41G/9.98G [00:07<00:49, 174MB/s]
pytorch_model-00001-of-00002.bin:  14% 1.43G/9.98G [00:08<00:48, 176MB/s]
pytorch_model-00001-of-00002.bin:  15% 1.45G/9.98G [00:08<00:47, 179MB/s]
pytorch_model-00001-of-00002.bin:  15% 1.47G/9.98G [00:08<00:47, 180MB/s]
pytorch_model-00001-of-00002.bin:  15% 1.49G/9.98G [00:08<00:46, 181MB/s]
pytorch_model-00001-of-00002.bin:  15% 1.51G/9.98G [00:08<00:46, 181MB/s]
pytorch_model-00001-of-00002.bin:  15% 1.53G/9.98G [00:08<00:46, 182MB/s]
pytorch_model-00001-of-00002.bin:  16% 1.55G/9.98G [00:08<00:48, 175MB/s]
pytorch_model-00001-of-00002.bin:  16% 1.57G/9.98G [00:08<00:47, 178MB/s]
pytorch_model-00001-of-00002.bin:  16% 1.59G/9.98G [00:09<00:48, 173MB/s]
pytorch_model-00001-of-00002.bin:  16% 1.61G/9.98G [00:09<00:46, 178MB/s]
pytorch_model-00001-of-00002.bin:  16% 1.64G/9.98G [00:09<00:46, 179MB/s]
pytorch_model-00001-of-00002.bin:  17% 1.66G/9.98G [00:09<00:45, 181MB/s]
pytorch_model-00001-of-00002.bin:  17% 1.68G/9.98G [00:09<00:45, 181MB/s]
pytorch_model-00001-of-00002.bin:  17% 1.70G/9.98G [00:09<00:45, 182MB/s]
pytorch_model-00001-of-00002.bin:  17% 1.72G/9.98G [00:09<00:44, 183MB/s]
pytorch_model-00001-of-00002.bin:  17% 1.74G/9.98G [00:09<00:44, 184MB/s]
pytorch_model-00001-of-00002.bin:  18% 1.76G/9.98G [00:09<00:44, 184MB/s]
pytorch_model-00001-of-00002.bin:  18% 1.78G/9.98G [00:10<00:44, 186MB/s]
pytorch_model-00001-of-00002.bin:  18% 1.80G/9.98G [00:10<00:54, 150MB/s]
pytorch_model-00001-of-00002.bin:  18% 1.82G/9.98G [00:10<01:04, 127MB/s]
pytorch_model-00001-of-00002.bin:  18% 1.85G/9.98G [00:10<01:07, 120MB/s]
pytorch_model-00001-of-00002.bin:  19% 1.87G/9.98G [00:10<01:10, 114MB/s]
pytorch_model-00001-of-00002.bin:  19% 1.89G/9.98G [00:11<01:10, 115MB/s]
pytorch_model-00001-of-00002.bin:  19% 1.91G/9.98G [00:11<01:17, 104MB/s]
pytorch_model-00001-of-00002.bin:  19% 1.93G/9.98G [00:11<01:26, 92.8MB/s]
pytorch_model-00001-of-00002.bin:  19% 1.94G/9.98G [00:11<01:30, 89.0MB/s]
pytorch_model-00001-of-00002.bin:  20% 1.95G/9.98G [00:11<01:28, 90.6MB/s]
pytorch_model-00001-of-00002.bin:  20% 1.96G/9.98G [00:11<01:34, 85.0MB/s]
pytorch_model-00001-of-00002.bin:  20% 1.97G/9.98G [00:12<01:33, 85.7MB/s]
pytorch_model-00001-of-00002.bin:  20% 1.98G/9.98G [00:12<01:36, 83.0MB/s]
pytorch_model-00001-of-00002.bin:  20% 1.99G/9.98G [00:12<01:40, 79.1MB/s]
pytorch_model-00001-of-00002.bin:  20% 2.00G/9.98G [00:12<01:42, 78.1MB/s]
pytorch_model-00001-of-00002.bin:  20% 2.01G/9.98G [00:12<01:39, 80.3MB/s]
pytorch_model-00001-of-00002.bin:  20% 2.02G/9.98G [00:12<01:44, 76.1MB/s]
pytorch_model-00001-of-00002.bin:  20% 2.04G/9.98G [00:12<01:29, 88.7MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.07G/9.98G [00:13<01:31, 86.8MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.08G/9.98G [00:13<01:27, 90.0MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.09G/9.98G [00:13<01:27, 90.4MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.10G/9.98G [00:13<01:29, 87.7MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.11G/9.98G [00:13<01:37, 80.3MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.12G/9.98G [00:13<01:34, 83.6MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.13G/9.98G [00:13<01:38, 79.9MB/s]
pytorch_model-00001-of-00002.bin:  21% 2.14G/9.98G [00:14<01:33, 83.5MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.15G/9.98G [00:14<01:30, 86.8MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.16G/9.98G [00:14<01:36, 81.3MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.17G/9.98G [00:14<01:31, 85.4MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.18G/9.98G [00:14<01:36, 81.1MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.19G/9.98G [00:14<01:29, 86.6MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.21G/9.98G [00:14<01:29, 86.7MB/s]
pytorch_model-00001-of-00002.bin:  22% 2.23G/9.98G [00:15<01:12, 106MB/s] 
pytorch_model-00001-of-00002.bin:  23% 2.25G/9.98G [00:15<01:02, 124MB/s]
pytorch_model-00001-of-00002.bin:  23% 2.28G/9.98G [00:15<00:55, 140MB/s]
pytorch_model-00001-of-00002.bin:  23% 2.30G/9.98G [00:15<00:50, 152MB/s]
pytorch_model-00001-of-00002.bin:  23% 2.32G/9.98G [00:15<00:49, 155MB/s]
pytorch_model-00001-of-00002.bin:  23% 2.34G/9.98G [00:15<00:50, 152MB/s]
pytorch_model-00001-of-00002.bin:  24% 2.36G/9.98G [00:15<00:46, 164MB/s]
pytorch_model-00001-of-00002.bin:  24% 2.38G/9.98G [00:15<00:45, 169MB/s]
pytorch_model-00001-of-00002.bin:  24% 2.40G/9.98G [00:16<00:43, 174MB/s]
pytorch_model-00001-of-00002.bin:  24% 2.42G/9.98G [00:16<00:48, 155MB/s]
pytorch_model-00001-of-00002.bin:  24% 2.44G/9.98G [00:16<00:45, 164MB/s]
pytorch_model-00001-of-00002.bin:  25% 2.46G/9.98G [00:16<00:45, 164MB/s]
pytorch_model-00001-of-00002.bin:  25% 2.49G/9.98G [00:16<00:44, 168MB/s]
pytorch_model-00001-of-00002.bin:  25% 2.51G/9.98G [00:16<00:42, 174MB/s]
pytorch_model-00001-of-00002.bin:  25% 2.53G/9.98G [00:16<00:42, 175MB/s]
pytorch_model-00001-of-00002.bin:  26% 2.55G/9.98G [00:16<00:41, 178MB/s]
pytorch_model-00001-of-00002.bin:  26% 2.57G/9.98G [00:17<00:41, 177MB/s]
pytorch_model-00001-of-00002.bin:  26% 2.59G/9.98G [00:17<00:41, 179MB/s]
pytorch_model-00001-of-00002.bin:  26% 2.61G/9.98G [00:17<00:40, 180MB/s]
pytorch_model-00001-of-00002.bin:  26% 2.63G/9.98G [00:17<00:40, 182MB/s]
pytorch_model-00001-of-00002.bin:  27% 2.65G/9.98G [00:17<00:39, 183MB/s]
pytorch_model-00001-of-00002.bin:  27% 2.67G/9.98G [00:17<00:39, 184MB/s]
pytorch_model-00001-of-00002.bin:  27% 2.69G/9.98G [00:17<00:39, 185MB/s]
pytorch_model-00001-of-00002.bin:  27% 2.72G/9.98G [00:17<00:41, 177MB/s]
pytorch_model-00001-of-00002.bin:  27% 2.74G/9.98G [00:20<04:41, 25.7MB/s]
pytorch_model-00001-of-00002.bin:  28% 2.77G/9.98G [00:20<03:03, 39.2MB/s]
pytorch_model-00001-of-00002.bin:  28% 2.79G/9.98G [00:20<02:24, 49.8MB/s]
pytorch_model-00001-of-00002.bin:  28% 2.81G/9.98G [00:20<01:54, 62.6MB/s]
pytorch_model-00001-of-00002.bin:  28% 2.83G/9.98G [00:20<01:32, 77.6MB/s]
pytorch_model-00001-of-00002.bin:  29% 2.85G/9.98G [00:20<01:16, 92.9MB/s]
pytorch_model-00001-of-00002.bin:  29% 2.87G/9.98G [00:20<01:05, 108MB/s] 
pytorch_model-00001-of-00002.bin:  29% 2.89G/9.98G [00:21<00:57, 123MB/s]
pytorch_model-00001-of-00002.bin:  29% 2.92G/9.98G [00:21<00:51, 137MB/s]
pytorch_model-00001-of-00002.bin:  29% 2.94G/9.98G [00:21<00:48, 146MB/s]
pytorch_model-00001-of-00002.bin:  30% 2.96G/9.98G [00:21<00:44, 159MB/s]
pytorch_model-00001-of-00002.bin:  30% 2.98G/9.98G [00:21<00:42, 163MB/s]
pytorch_model-00001-of-00002.bin:  30% 3.00G/9.98G [00:21<00:42, 166MB/s]
pytorch_model-00001-of-00002.bin:  30% 3.02G/9.98G [00:21<00:40, 174MB/s]
pytorch_model-00001-of-00002.bin:  30% 3.04G/9.98G [00:21<00:39, 174MB/s]
pytorch_model-00001-of-00002.bin:  31% 3.06G/9.98G [00:22<00:38, 178MB/s]
pytorch_model-00001-of-00002.bin:  31% 3.08G/9.98G [00:22<00:38, 178MB/s]
pytorch_model-00001-of-00002.bin:  31% 3.10G/9.98G [00:22<00:37, 182MB/s]
pytorch_model-00001-of-00002.bin:  31% 3.12G/9.98G [00:22<00:37, 182MB/s]
pytorch_model-00001-of-00002.bin:  32% 3.15G/9.98G [00:22<00:38, 177MB/s]
pytorch_model-00001-of-00002.bin:  32% 3.17G/9.98G [00:22<00:38, 179MB/s]
pytorch_model-00001-of-00002.bin:  32% 3.19G/9.98G [00:22<00:37, 180MB/s]
pytorch_model-00001-of-00002.bin:  32% 3.21G/9.98G [00:22<00:37, 181MB/s]
pytorch_model-00001-of-00002.bin:  32% 3.23G/9.98G [00:22<00:37, 182MB/s]
pytorch_model-00001-of-00002.bin:  33% 3.25G/9.98G [00:23<00:36, 183MB/s]
pytorch_model-00001-of-00002.bin:  33% 3.27G/9.98G [00:23<00:36, 182MB/s]
pytorch_model-00001-of-00002.bin:  33% 3.29G/9.98G [00:23<00:36, 182MB/s]
pytorch_model-00001-of-00002.bin:  33% 3.31G/9.98G [00:23<00:36, 183MB/s]
pytorch_model-00001-of-00002.bin:  33% 3.33G/9.98G [00:23<00:36, 184MB/s]
pytorch_model-00001-of-00002.bin:  34% 3.36G/9.98G [00:23<00:36, 184MB/s]
pytorch_model-00001-of-00002.bin:  34% 3.38G/9.98G [00:23<00:36, 183MB/s]
pytorch_model-00001-of-00002.bin:  34% 3.40G/9.98G [00:23<00:35, 184MB/s]
pytorch_model-00001-of-00002.bin:  34% 3.42G/9.98G [00:23<00:35, 184MB/s]
pytorch_model-00001-of-00002.bin:  34% 3.44G/9.98G [00:24<00:35, 185MB/s]
pytorch_model-00001-of-00002.bin:  35% 3.46G/9.98G [00:24<00:36, 176MB/s]
pytorch_model-00001-of-00002.bin:  35% 3.48G/9.98G [00:24<00:36, 179MB/s]
pytorch_model-00001-of-00002.bin:  35% 3.50G/9.98G [00:24<00:34, 187MB/s]
pytorch_model-00001-of-00002.bin:  35% 3.52G/9.98G [00:24<00:34, 186MB/s]
pytorch_model-00001-of-00002.bin:  36% 3.54G/9.98G [00:24<00:34, 186MB/s]
pytorch_model-00001-of-00002.bin:  36% 3.57G/9.98G [00:24<00:34, 186MB/s]
pytorch_model-00001-of-00002.bin:  36% 3.59G/9.98G [00:24<00:34, 187MB/s]
pytorch_model-00001-of-00002.bin:  36% 3.61G/9.98G [00:24<00:34, 185MB/s]
pytorch_model-00001-of-00002.bin:  36% 3.63G/9.98G [00:25<00:34, 185MB/s]
pytorch_model-00001-of-00002.bin:  37% 3.65G/9.98G [00:25<00:47, 134MB/s]
pytorch_model-00001-of-00002.bin:  37% 3.67G/9.98G [00:25<00:49, 128MB/s]
pytorch_model-00001-of-00002.bin:  37% 3.69G/9.98G [00:25<00:54, 116MB/s]
pytorch_model-00001-of-00002.bin:  37% 3.71G/9.98G [00:25<00:51, 121MB/s]
pytorch_model-00001-of-00002.bin:  37% 3.73G/9.98G [00:26<00:54, 114MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.75G/9.98G [00:26<00:56, 110MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.77G/9.98G [00:26<01:02, 100MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.80G/9.98G [00:26<01:05, 94.9MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.82G/9.98G [00:27<01:07, 91.4MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.83G/9.98G [00:27<01:06, 91.9MB/s]
pytorch_model-00001-of-00002.bin:  38% 3.84G/9.98G [00:27<01:05, 93.8MB/s]
pytorch_model-00001-of-00002.bin:  39% 3.86G/9.98G [00:27<01:03, 95.9MB/s]
pytorch_model-00001-of-00002.bin:  39% 3.88G/9.98G [00:27<01:00, 102MB/s] 
pytorch_model-00001-of-00002.bin:  39% 3.89G/9.98G [00:27<01:05, 92.3MB/s]
pytorch_model-00001-of-00002.bin:  39% 3.91G/9.98G [00:28<00:58, 104MB/s] 
pytorch_model-00001-of-00002.bin:  39% 3.93G/9.98G [00:28<00:57, 105MB/s]
pytorch_model-00001-of-00002.bin:  40% 3.95G/9.98G [00:28<01:01, 98.6MB/s]
pytorch_model-00001-of-00002.bin:  40% 3.96G/9.98G [00:28<01:01, 98.0MB/s]
pytorch_model-00001-of-00002.bin:  40% 3.97G/9.98G [00:28<01:05, 91.8MB/s]
pytorch_model-00001-of-00002.bin:  40% 3.98G/9.98G [00:28<01:09, 86.2MB/s]
pytorch_model-00001-of-00002.bin:  40% 4.00G/9.98G [00:28<01:13, 81.6MB/s]
pytorch_model-00001-of-00002.bin:  40% 4.02G/9.98G [00:29<00:57, 104MB/s] 
pytorch_model-00001-of-00002.bin:  40% 4.04G/9.98G [00:29<00:47, 125MB/s]
pytorch_model-00001-of-00002.bin:  41% 4.06G/9.98G [00:29<00:42, 140MB/s]
pytorch_model-00001-of-00002.bin:  41% 4.08G/9.98G [00:29<00:39, 151MB/s]
pytorch_model-00001-of-00002.bin:  41% 4.10G/9.98G [00:29<00:36, 160MB/s]
pytorch_model-00001-of-00002.bin:  41% 4.12G/9.98G [00:29<00:35, 165MB/s]
pytorch_model-00001-of-00002.bin:  42% 4.14G/9.98G [00:29<00:33, 173MB/s]
pytorch_model-00001-of-00002.bin:  42% 4.16G/9.98G [00:29<00:32, 177MB/s]
pytorch_model-00001-of-00002.bin:  42% 4.18G/9.98G [00:30<00:32, 179MB/s]
pytorch_model-00001-of-00002.bin:  42% 4.20G/9.98G [00:30<00:31, 181MB/s]
pytorch_model-00001-of-00002.bin:  42% 4.23G/9.98G [00:30<00:31, 182MB/s]
pytorch_model-00001-of-00002.bin:  43% 4.25G/9.98G [00:30<00:33, 170MB/s]
pytorch_model-00001-of-00002.bin:  43% 4.27G/9.98G [00:30<00:31, 180MB/s]
pytorch_model-00001-of-00002.bin:  43% 4.29G/9.98G [00:30<00:32, 176MB/s]
pytorch_model-00001-of-00002.bin:  43% 4.31G/9.98G [00:30<00:48, 118MB/s]
pytorch_model-00001-of-00002.bin:  44% 4.34G/9.98G [00:31<00:37, 149MB/s]
pytorch_model-00001-of-00002.bin:  44% 4.36G/9.98G [00:31<00:36, 155MB/s]
pytorch_model-00001-of-00002.bin:  44% 4.38G/9.98G [00:31<00:34, 162MB/s]
pytorch_model-00001-of-00002.bin:  44% 4.40G/9.98G [00:31<00:33, 165MB/s]
pytorch_model-00001-of-00002.bin:  44% 4.42G/9.98G [00:31<00:32, 171MB/s]
pytorch_model-00001-of-00002.bin:  45% 4.45G/9.98G [00:31<00:31, 176MB/s]
pytorch_model-00001-of-00002.bin:  45% 4.47G/9.98G [00:31<00:31, 177MB/s]
pytorch_model-00001-of-00002.bin:  45% 4.49G/9.98G [00:31<00:30, 179MB/s]
pytorch_model-00001-of-00002.bin:  45% 4.51G/9.98G [00:31<00:30, 182MB/s]
pytorch_model-00001-of-00002.bin:  45% 4.53G/9.98G [00:32<00:29, 183MB/s]
pytorch_model-00001-of-00002.bin:  46% 4.55G/9.98G [00:32<00:29, 184MB/s]
pytorch_model-00001-of-00002.bin:  46% 4.57G/9.98G [00:32<00:29, 183MB/s]
pytorch_model-00001-of-00002.bin:  46% 4.59G/9.98G [00:32<00:29, 183MB/s]
pytorch_model-00001-of-00002.bin:  46% 4.61G/9.98G [00:32<00:29, 182MB/s]
pytorch_model-00001-of-00002.bin:  46% 4.63G/9.98G [00:32<00:29, 183MB/s]
pytorch_model-00001-of-00002.bin:  47% 4.66G/9.98G [00:32<00:29, 182MB/s]
pytorch_model-00001-of-00002.bin:  47% 4.68G/9.98G [00:32<00:28, 184MB/s]
pytorch_model-00001-of-00002.bin:  47% 4.70G/9.98G [00:33<00:28, 184MB/s]
pytorch_model-00001-of-00002.bin:  47% 4.72G/9.98G [00:33<00:28, 185MB/s]
pytorch_model-00001-of-00002.bin:  48% 4.74G/9.98G [00:33<00:28, 184MB/s]
pytorch_model-00001-of-00002.bin:  48% 4.76G/9.98G [00:33<00:28, 185MB/s]
pytorch_model-00001-of-00002.bin:  48% 4.78G/9.98G [00:33<00:28, 184MB/s]
pytorch_model-00001-of-00002.bin:  48% 4.80G/9.98G [00:33<00:28, 184MB/s]
pytorch_model-00001-of-00002.bin:  48% 4.82G/9.98G [00:33<00:28, 182MB/s]
pytorch_model-00001-of-00002.bin:  49% 4.84G/9.98G [00:33<00:28, 183MB/s]
pytorch_model-00001-of-00002.bin:  49% 4.87G/9.98G [00:33<00:27, 184MB/s]
pytorch_model-00001-of-00002.bin:  49% 4.89G/9.98G [00:34<00:27, 182MB/s]
pytorch_model-00001-of-00002.bin:  49% 4.91G/9.98G [00:34<00:27, 184MB/s]
pytorch_model-00001-of-00002.bin:  49% 4.93G/9.98G [00:34<00:27, 183MB/s]
pytorch_model-00001-of-00002.bin:  50% 4.95G/9.98G [00:34<00:27, 183MB/s]
pytorch_model-00001-of-00002.bin:  50% 4.97G/9.98G [00:34<00:27, 184MB/s]
pytorch_model-00001-of-00002.bin:  50% 4.99G/9.98G [00:34<00:27, 184MB/s]
pytorch_model-00001-of-00002.bin:  50% 5.01G/9.98G [00:34<00:26, 185MB/s]
pytorch_model-00001-of-00002.bin:  50% 5.03G/9.98G [00:34<00:27, 182MB/s]
pytorch_model-00001-of-00002.bin:  51% 5.05G/9.98G [00:34<00:27, 181MB/s]
pytorch_model-00001-of-00002.bin:  51% 5.08G/9.98G [00:35<00:26, 182MB/s]
pytorch_model-00001-of-00002.bin:  51% 5.10G/9.98G [00:35<00:26, 182MB/s]
pytorch_model-00001-of-00002.bin:  51% 5.12G/9.98G [00:35<00:26, 182MB/s]
pytorch_model-00001-of-00002.bin:  52% 5.14G/9.98G [00:35<00:26, 181MB/s]
pytorch_model-00001-of-00002.bin:  52% 5.16G/9.98G [00:35<00:26, 181MB/s]
pytorch_model-00001-of-00002.bin:  52% 5.18G/9.98G [00:35<00:26, 183MB/s]
pytorch_model-00001-of-00002.bin:  52% 5.20G/9.98G [00:35<00:25, 184MB/s]
pytorch_model-00001-of-00002.bin:  52% 5.22G/9.98G [00:35<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  53% 5.24G/9.98G [00:35<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  53% 5.26G/9.98G [00:36<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  53% 5.28G/9.98G [00:36<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  53% 5.31G/9.98G [00:36<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  53% 5.33G/9.98G [00:36<00:25, 185MB/s]
pytorch_model-00001-of-00002.bin:  54% 5.35G/9.98G [00:36<00:25, 181MB/s]
pytorch_model-00001-of-00002.bin:  54% 5.37G/9.98G [00:36<00:25, 183MB/s]
pytorch_model-00001-of-00002.bin:  54% 5.39G/9.98G [00:36<00:24, 184MB/s]
pytorch_model-00001-of-00002.bin:  54% 5.41G/9.98G [00:36<00:24, 183MB/s]
pytorch_model-00001-of-00002.bin:  54% 5.43G/9.98G [00:37<00:24, 185MB/s]
pytorch_model-00001-of-00002.bin:  55% 5.45G/9.98G [00:37<00:24, 185MB/s]
pytorch_model-00001-of-00002.bin:  55% 5.47G/9.98G [00:37<00:24, 185MB/s]
pytorch_model-00001-of-00002.bin:  55% 5.49G/9.98G [00:37<00:24, 185MB/s]
pytorch_model-00001-of-00002.bin:  55% 5.52G/9.98G [00:37<00:24, 183MB/s]
pytorch_model-00001-of-00002.bin:  55% 5.54G/9.98G [00:37<00:24, 183MB/s]
pytorch_model-00001-of-00002.bin:  56% 5.56G/9.98G [00:37<00:24, 184MB/s]
pytorch_model-00001-of-00002.bin:  56% 5.58G/9.98G [00:37<00:23, 185MB/s]
pytorch_model-00001-of-00002.bin:  56% 5.60G/9.98G [00:37<00:23, 185MB/s]
pytorch_model-00001-of-00002.bin:  56% 5.62G/9.98G [00:38<00:23, 186MB/s]
pytorch_model-00001-of-00002.bin:  57% 5.64G/9.98G [00:38<00:23, 182MB/s]
pytorch_model-00001-of-00002.bin:  57% 5.66G/9.98G [00:38<00:23, 183MB/s]
pytorch_model-00001-of-00002.bin:  57% 5.68G/9.98G [00:38<00:23, 182MB/s]
pytorch_model-00001-of-00002.bin:  57% 5.70G/9.98G [00:38<00:23, 181MB/s]
pytorch_model-00001-of-00002.bin:  57% 5.73G/9.98G [00:38<00:23, 182MB/s]
pytorch_model-00001-of-00002.bin:  58% 5.75G/9.98G [00:38<00:23, 181MB/s]
pytorch_model-00001-of-00002.bin:  58% 5.77G/9.98G [00:38<00:23, 182MB/s]
pytorch_model-00001-of-00002.bin:  58% 5.79G/9.98G [00:38<00:23, 181MB/s]
pytorch_model-00001-of-00002.bin:  58% 5.81G/9.98G [00:39<00:22, 182MB/s]
pytorch_model-00001-of-00002.bin:  58% 5.83G/9.98G [00:39<00:30, 137MB/s]
pytorch_model-00001-of-00002.bin:  59% 5.85G/9.98G [00:39<00:32, 126MB/s]
pytorch_model-00001-of-00002.bin:  59% 5.87G/9.98G [00:39<00:34, 119MB/s]
pytorch_model-00001-of-00002.bin:  59% 5.89G/9.98G [00:39<00:31, 130MB/s]
pytorch_model-00001-of-00002.bin:  59% 5.91G/9.98G [00:39<00:28, 142MB/s]
pytorch_model-00001-of-00002.bin:  59% 5.93G/9.98G [00:40<00:32, 125MB/s]
pytorch_model-00001-of-00002.bin:  60% 5.96G/9.98G [00:40<00:39, 101MB/s]
pytorch_model-00001-of-00002.bin:  60% 5.98G/9.98G [00:40<00:39, 102MB/s]
pytorch_model-00001-of-00002.bin:  60% 6.00G/9.98G [00:40<00:39, 101MB/s]
pytorch_model-00001-of-00002.bin:  60% 6.02G/9.98G [00:41<00:43, 91.2MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.04G/9.98G [00:41<00:37, 104MB/s] 
pytorch_model-00001-of-00002.bin:  61% 6.06G/9.98G [00:41<00:41, 94.2MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.07G/9.98G [00:41<00:43, 90.6MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.08G/9.98G [00:41<00:44, 86.7MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.09G/9.98G [00:42<00:47, 81.7MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.10G/9.98G [00:42<00:46, 83.9MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.11G/9.98G [00:42<00:47, 80.5MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.12G/9.98G [00:42<00:49, 77.8MB/s]
pytorch_model-00001-of-00002.bin:  61% 6.13G/9.98G [00:42<00:50, 75.6MB/s]
pytorch_model-00001-of-00002.bin:  62% 6.16G/9.98G [00:42<00:44, 85.6MB/s]
pytorch_model-00001-of-00002.bin:  62% 6.18G/9.98G [00:42<00:41, 90.9MB/s]
pytorch_model-00001-of-00002.bin:  62% 6.20G/9.98G [00:43<00:34, 110MB/s] 
pytorch_model-00001-of-00002.bin:  62% 6.22G/9.98G [00:43<00:30, 123MB/s]
pytorch_model-00001-of-00002.bin:  63% 6.24G/9.98G [00:43<00:37, 99.5MB/s]
pytorch_model-00001-of-00002.bin:  63% 6.26G/9.98G [00:43<00:38, 97.2MB/s]
pytorch_model-00001-of-00002.bin:  63% 6.28G/9.98G [00:43<00:37, 99.4MB/s]
pytorch_model-00001-of-00002.bin:  63% 6.30G/9.98G [00:44<00:37, 97.6MB/s]
pytorch_model-00001-of-00002.bin:  63% 6.32G/9.98G [00:44<00:32, 114MB/s] 
pytorch_model-00001-of-00002.bin:  64% 6.34G/9.98G [00:44<00:29, 125MB/s]
pytorch_model-00001-of-00002.bin:  64% 6.36G/9.98G [00:44<00:25, 139MB/s]
pytorch_model-00001-of-00002.bin:  64% 6.39G/9.98G [00:44<00:23, 151MB/s]
pytorch_model-00001-of-00002.bin:  64% 6.41G/9.98G [00:44<00:22, 160MB/s]
pytorch_model-00001-of-00002.bin:  64% 6.43G/9.98G [00:44<00:21, 164MB/s]
pytorch_model-00001-of-00002.bin:  65% 6.45G/9.98G [00:44<00:20, 170MB/s]
pytorch_model-00001-of-00002.bin:  65% 6.47G/9.98G [00:45<00:20, 174MB/s]
pytorch_model-00001-of-00002.bin:  65% 6.49G/9.98G [00:45<00:19, 178MB/s]
pytorch_model-00001-of-00002.bin:  65% 6.51G/9.98G [00:45<00:19, 180MB/s]
pytorch_model-00001-of-00002.bin:  65% 6.53G/9.98G [00:45<00:19, 177MB/s]
pytorch_model-00001-of-00002.bin:  66% 6.55G/9.98G [00:45<00:19, 178MB/s]
pytorch_model-00001-of-00002.bin:  66% 6.57G/9.98G [00:45<00:19, 178MB/s]
pytorch_model-00001-of-00002.bin:  66% 6.60G/9.98G [00:45<00:18, 180MB/s]
pytorch_model-00001-of-00002.bin:  66% 6.62G/9.98G [00:45<00:18, 181MB/s]
pytorch_model-00001-of-00002.bin:  67% 6.64G/9.98G [00:46<00:18, 179MB/s]
pytorch_model-00001-of-00002.bin:  67% 6.66G/9.98G [00:46<00:18, 179MB/s]
pytorch_model-00001-of-00002.bin:  67% 6.68G/9.98G [00:46<00:18, 181MB/s]
pytorch_model-00001-of-00002.bin:  67% 6.70G/9.98G [00:46<00:17, 182MB/s]
pytorch_model-00001-of-00002.bin:  67% 6.72G/9.98G [00:46<00:17, 183MB/s]
pytorch_model-00001-of-00002.bin:  68% 6.74G/9.98G [00:46<00:28, 115MB/s]
pytorch_model-00001-of-00002.bin:  68% 6.77G/9.98G [00:46<00:22, 143MB/s]
pytorch_model-00001-of-00002.bin:  68% 6.81G/9.98G [00:47<00:19, 159MB/s]
pytorch_model-00001-of-00002.bin:  68% 6.83G/9.98G [00:47<00:19, 165MB/s]
pytorch_model-00001-of-00002.bin:  69% 6.85G/9.98G [00:47<00:19, 164MB/s]
pytorch_model-00001-of-00002.bin:  69% 6.87G/9.98G [00:47<00:19, 163MB/s]
pytorch_model-00001-of-00002.bin:  69% 6.89G/9.98G [00:47<00:18, 168MB/s]
pytorch_model-00001-of-00002.bin:  69% 6.91G/9.98G [00:47<00:17, 172MB/s]
pytorch_model-00001-of-00002.bin:  69% 6.93G/9.98G [00:47<00:17, 176MB/s]
pytorch_model-00001-of-00002.bin:  70% 6.95G/9.98G [00:47<00:16, 179MB/s]
pytorch_model-00001-of-00002.bin:  70% 6.97G/9.98G [00:48<00:16, 181MB/s]
pytorch_model-00001-of-00002.bin:  70% 6.99G/9.98G [00:48<00:16, 181MB/s]
pytorch_model-00001-of-00002.bin:  70% 7.01G/9.98G [00:48<00:16, 182MB/s]
pytorch_model-00001-of-00002.bin:  71% 7.04G/9.98G [00:48<00:16, 182MB/s]
pytorch_model-00001-of-00002.bin:  71% 7.06G/9.98G [00:48<00:15, 183MB/s]
pytorch_model-00001-of-00002.bin:  71% 7.08G/9.98G [00:48<00:15, 184MB/s]
pytorch_model-00001-of-00002.bin:  71% 7.10G/9.98G [00:48<00:15, 183MB/s]
pytorch_model-00001-of-00002.bin:  71% 7.12G/9.98G [00:48<00:16, 178MB/s]
pytorch_model-00001-of-00002.bin:  72% 7.14G/9.98G [00:48<00:15, 178MB/s]
pytorch_model-00001-of-00002.bin:  72% 7.16G/9.98G [00:49<00:15, 180MB/s]
pytorch_model-00001-of-00002.bin:  72% 7.18G/9.98G [00:49<00:15, 182MB/s]
pytorch_model-00001-of-00002.bin:  72% 7.20G/9.98G [00:49<00:15, 182MB/s]
pytorch_model-00001-of-00002.bin:  72% 7.22G/9.98G [00:49<00:15, 180MB/s]
pytorch_model-00001-of-00002.bin:  73% 7.25G/9.98G [00:49<00:15, 182MB/s]
pytorch_model-00001-of-00002.bin:  73% 7.27G/9.98G [00:49<00:14, 183MB/s]
pytorch_model-00001-of-00002.bin:  73% 7.29G/9.98G [00:49<00:14, 183MB/s]
pytorch_model-00001-of-00002.bin:  73% 7.31G/9.98G [00:49<00:14, 184MB/s]
pytorch_model-00001-of-00002.bin:  73% 7.33G/9.98G [00:50<00:14, 184MB/s]
pytorch_model-00001-of-00002.bin:  74% 7.35G/9.98G [00:50<00:14, 179MB/s]
pytorch_model-00001-of-00002.bin:  74% 7.37G/9.98G [00:50<00:14, 183MB/s]
pytorch_model-00001-of-00002.bin:  74% 7.39G/9.98G [00:50<00:14, 183MB/s]
pytorch_model-00001-of-00002.bin:  74% 7.41G/9.98G [00:50<00:14, 183MB/s]
pytorch_model-00001-of-00002.bin:  75% 7.43G/9.98G [00:50<00:13, 184MB/s]
pytorch_model-00001-of-00002.bin:  75% 7.46G/9.98G [00:50<00:13, 183MB/s]
pytorch_model-00001-of-00002.bin:  75% 7.48G/9.98G [00:50<00:13, 183MB/s]
pytorch_model-00001-of-00002.bin:  75% 7.50G/9.98G [00:50<00:13, 185MB/s]
pytorch_model-00001-of-00002.bin:  75% 7.52G/9.98G [00:51<00:13, 185MB/s]
pytorch_model-00001-of-00002.bin:  76% 7.54G/9.98G [00:51<00:13, 180MB/s]
pytorch_model-00001-of-00002.bin:  76% 7.56G/9.98G [00:51<00:13, 181MB/s]
pytorch_model-00001-of-00002.bin:  76% 7.58G/9.98G [00:51<00:13, 181MB/s]
pytorch_model-00001-of-00002.bin:  76% 7.60G/9.98G [00:51<00:13, 182MB/s]
pytorch_model-00001-of-00002.bin:  76% 7.62G/9.98G [00:51<00:12, 184MB/s]
pytorch_model-00001-of-00002.bin:  77% 7.64G/9.98G [00:51<00:12, 183MB/s]
pytorch_model-00001-of-00002.bin:  77% 7.67G/9.98G [00:51<00:12, 182MB/s]
pytorch_model-00001-of-00002.bin:  77% 7.69G/9.98G [00:51<00:12, 182MB/s]
pytorch_model-00001-of-00002.bin:  77% 7.71G/9.98G [00:52<00:12, 183MB/s]
pytorch_model-00001-of-00002.bin:  77% 7.73G/9.98G [00:52<00:12, 183MB/s]
pytorch_model-00001-of-00002.bin:  78% 7.75G/9.98G [00:52<00:12, 184MB/s]
pytorch_model-00001-of-00002.bin:  78% 7.77G/9.98G [00:52<00:12, 177MB/s]
pytorch_model-00001-of-00002.bin:  78% 7.79G/9.98G [00:52<00:12, 180MB/s]
pytorch_model-00001-of-00002.bin:  78% 7.81G/9.98G [00:52<00:12, 180MB/s]
pytorch_model-00001-of-00002.bin:  79% 7.83G/9.98G [00:52<00:11, 182MB/s]
pytorch_model-00001-of-00002.bin:  79% 7.85G/9.98G [00:52<00:11, 178MB/s]
pytorch_model-00001-of-00002.bin:  79% 7.87G/9.98G [00:53<00:11, 180MB/s]
pytorch_model-00001-of-00002.bin:  79% 7.90G/9.98G [00:53<00:11, 182MB/s]
pytorch_model-00001-of-00002.bin:  79% 7.92G/9.98G [00:53<00:11, 183MB/s]
pytorch_model-00001-of-00002.bin:  80% 7.94G/9.98G [00:53<00:11, 183MB/s]
pytorch_model-00001-of-00002.bin:  80% 7.96G/9.98G [00:53<00:10, 184MB/s]
pytorch_model-00001-of-00002.bin:  80% 7.98G/9.98G [00:53<00:10, 184MB/s]
pytorch_model-00001-of-00002.bin:  80% 8.00G/9.98G [00:53<00:10, 185MB/s]
pytorch_model-00001-of-00002.bin:  80% 8.02G/9.98G [00:53<00:10, 184MB/s]
pytorch_model-00001-of-00002.bin:  81% 8.04G/9.98G [00:53<00:10, 182MB/s]
pytorch_model-00001-of-00002.bin:  81% 8.06G/9.98G [00:54<00:10, 185MB/s]
pytorch_model-00001-of-00002.bin:  81% 8.08G/9.98G [00:54<00:10, 187MB/s]
pytorch_model-00001-of-00002.bin:  81% 8.11G/9.98G [00:54<00:10, 186MB/s]
pytorch_model-00001-of-00002.bin:  81% 8.13G/9.98G [00:54<00:12, 150MB/s]
pytorch_model-00001-of-00002.bin:  82% 8.15G/9.98G [00:54<00:16, 112MB/s]
pytorch_model-00001-of-00002.bin:  82% 8.17G/9.98G [00:54<00:16, 108MB/s]
pytorch_model-00001-of-00002.bin:  82% 8.19G/9.98G [00:55<00:16, 106MB/s]
pytorch_model-00001-of-00002.bin:  82% 8.21G/9.98G [00:55<00:17, 100MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.23G/9.98G [00:55<00:20, 86.7MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.24G/9.98G [00:55<00:21, 82.3MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.25G/9.98G [00:56<00:24, 71.2MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.27G/9.98G [00:56<00:20, 82.1MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.28G/9.98G [00:56<00:20, 80.8MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.29G/9.98G [00:56<00:21, 77.5MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.32G/9.98G [00:56<00:19, 84.5MB/s]
pytorch_model-00001-of-00002.bin:  83% 8.33G/9.98G [00:56<00:19, 82.7MB/s]
pytorch_model-00001-of-00002.bin:  84% 8.34G/9.98G [00:57<00:20, 78.5MB/s]
pytorch_model-00001-of-00002.bin:  84% 8.36G/9.98G [00:57<00:16, 101MB/s] 
pytorch_model-00001-of-00002.bin:  84% 8.38G/9.98G [00:57<00:13, 118MB/s]
pytorch_model-00001-of-00002.bin:  84% 8.40G/9.98G [00:57<00:13, 114MB/s]
pytorch_model-00001-of-00002.bin:  84% 8.42G/9.98G [00:57<00:13, 112MB/s]
pytorch_model-00001-of-00002.bin:  85% 8.44G/9.98G [00:57<00:14, 107MB/s]
pytorch_model-00001-of-00002.bin:  85% 8.46G/9.98G [00:58<00:12, 119MB/s]
pytorch_model-00001-of-00002.bin:  85% 8.48G/9.98G [00:58<00:12, 115MB/s]
pytorch_model-00001-of-00002.bin:  85% 8.50G/9.98G [00:58<00:13, 108MB/s]
pytorch_model-00001-of-00002.bin:  85% 8.52G/9.98G [00:58<00:13, 104MB/s]
pytorch_model-00001-of-00002.bin:  86% 8.55G/9.98G [00:58<00:13, 104MB/s]
pytorch_model-00001-of-00002.bin:  86% 8.57G/9.98G [00:59<00:14, 98.4MB/s]
pytorch_model-00001-of-00002.bin:  86% 8.59G/9.98G [00:59<00:12, 113MB/s] 
pytorch_model-00001-of-00002.bin:  86% 8.61G/9.98G [00:59<00:10, 127MB/s]
pytorch_model-00001-of-00002.bin:  86% 8.63G/9.98G [00:59<00:09, 140MB/s]
pytorch_model-00001-of-00002.bin:  87% 8.65G/9.98G [00:59<00:08, 152MB/s]
pytorch_model-00001-of-00002.bin:  87% 8.67G/9.98G [00:59<00:08, 160MB/s]
pytorch_model-00001-of-00002.bin:  87% 8.69G/9.98G [00:59<00:07, 167MB/s]
pytorch_model-00001-of-00002.bin:  87% 8.71G/9.98G [00:59<00:07, 173MB/s]
pytorch_model-00001-of-00002.bin:  88% 8.73G/9.98G [01:00<00:07, 176MB/s]
pytorch_model-00001-of-00002.bin:  88% 8.76G/9.98G [01:00<00:06, 175MB/s]
pytorch_model-00001-of-00002.bin:  88% 8.78G/9.98G [01:00<00:06, 177MB/s]
pytorch_model-00001-of-00002.bin:  88% 8.80G/9.98G [01:00<00:06, 180MB/s]
pytorch_model-00001-of-00002.bin:  88% 8.82G/9.98G [01:00<00:06, 182MB/s]
pytorch_model-00001-of-00002.bin:  89% 8.84G/9.98G [01:00<00:06, 182MB/s]
pytorch_model-00001-of-00002.bin:  89% 8.86G/9.98G [01:00<00:06, 180MB/s]
pytorch_model-00001-of-00002.bin:  89% 8.88G/9.98G [01:00<00:06, 182MB/s]
pytorch_model-00001-of-00002.bin:  89% 8.90G/9.98G [01:01<00:05, 184MB/s]
pytorch_model-00001-of-00002.bin:  89% 8.92G/9.98G [01:01<00:05, 184MB/s]
pytorch_model-00001-of-00002.bin:  90% 8.94G/9.98G [01:01<00:05, 184MB/s]
pytorch_model-00001-of-00002.bin:  90% 8.97G/9.98G [01:01<00:05, 184MB/s]
pytorch_model-00001-of-00002.bin:  90% 8.99G/9.98G [01:01<00:05, 184MB/s]
pytorch_model-00001-of-00002.bin:  90% 9.01G/9.98G [01:01<00:05, 185MB/s]
pytorch_model-00001-of-00002.bin:  90% 9.03G/9.98G [01:01<00:05, 185MB/s]
pytorch_model-00001-of-00002.bin:  91% 9.05G/9.98G [01:01<00:05, 185MB/s]
pytorch_model-00001-of-00002.bin:  91% 9.07G/9.98G [01:01<00:04, 185MB/s]
pytorch_model-00001-of-00002.bin:  91% 9.09G/9.98G [01:02<00:04, 184MB/s]
pytorch_model-00001-of-00002.bin:  91% 9.11G/9.98G [01:02<00:04, 185MB/s]
pytorch_model-00001-of-00002.bin:  92% 9.13G/9.98G [01:02<00:04, 184MB/s]
pytorch_model-00001-of-00002.bin:  92% 9.15G/9.98G [01:02<00:04, 184MB/s]
pytorch_model-00001-of-00002.bin:  92% 9.18G/9.98G [01:02<00:04, 181MB/s]
pytorch_model-00001-of-00002.bin:  92% 9.20G/9.98G [01:02<00:04, 183MB/s]
pytorch_model-00001-of-00002.bin:  92% 9.22G/9.98G [01:02<00:04, 181MB/s]
pytorch_model-00001-of-00002.bin:  93% 9.24G/9.98G [01:02<00:04, 180MB/s]
pytorch_model-00001-of-00002.bin:  93% 9.26G/9.98G [01:02<00:03, 183MB/s]
pytorch_model-00001-of-00002.bin:  93% 9.28G/9.98G [01:03<00:03, 184MB/s]
pytorch_model-00001-of-00002.bin:  93% 9.30G/9.98G [01:03<00:03, 184MB/s]
pytorch_model-00001-of-00002.bin:  93% 9.32G/9.98G [01:03<00:03, 185MB/s]
pytorch_model-00001-of-00002.bin:  94% 9.34G/9.98G [01:03<00:03, 183MB/s]
pytorch_model-00001-of-00002.bin:  94% 9.36G/9.98G [01:03<00:03, 184MB/s]
pytorch_model-00001-of-00002.bin:  94% 9.38G/9.98G [01:03<00:03, 179MB/s]
pytorch_model-00001-of-00002.bin:  94% 9.41G/9.98G [01:03<00:03, 180MB/s]
pytorch_model-00001-of-00002.bin:  94% 9.43G/9.98G [01:03<00:03, 182MB/s]
pytorch_model-00001-of-00002.bin:  95% 9.45G/9.98G [01:03<00:02, 181MB/s]
pytorch_model-00001-of-00002.bin:  95% 9.47G/9.98G [01:04<00:02, 178MB/s]
pytorch_model-00001-of-00002.bin:  95% 9.49G/9.98G [01:04<00:02, 180MB/s]
pytorch_model-00001-of-00002.bin:  95% 9.51G/9.98G [01:04<00:02, 178MB/s]
pytorch_model-00001-of-00002.bin:  96% 9.53G/9.98G [01:04<00:02, 181MB/s]
pytorch_model-00001-of-00002.bin:  96% 9.55G/9.98G [01:04<00:03, 119MB/s]
pytorch_model-00001-of-00002.bin:  96% 9.58G/9.98G [01:04<00:02, 146MB/s]
pytorch_model-00001-of-00002.bin:  96% 9.60G/9.98G [01:05<00:02, 149MB/s]
pytorch_model-00001-of-00002.bin:  96% 9.63G/9.98G [01:05<00:02, 155MB/s]
pytorch_model-00001-of-00002.bin:  97% 9.65G/9.98G [01:05<00:02, 163MB/s]
pytorch_model-00001-of-00002.bin:  97% 9.67G/9.98G [01:05<00:01, 169MB/s]
pytorch_model-00001-of-00002.bin:  97% 9.69G/9.98G [01:05<00:01, 174MB/s]
pytorch_model-00001-of-00002.bin:  97% 9.71G/9.98G [01:05<00:01, 176MB/s]
pytorch_model-00001-of-00002.bin:  98% 9.73G/9.98G [01:05<00:01, 178MB/s]
pytorch_model-00001-of-00002.bin:  98% 9.75G/9.98G [01:05<00:01, 174MB/s]
pytorch_model-00001-of-00002.bin:  98% 9.77G/9.98G [01:05<00:01, 177MB/s]
pytorch_model-00001-of-00002.bin:  98% 9.79G/9.98G [01:06<00:01, 179MB/s]
pytorch_model-00001-of-00002.bin:  98% 9.81G/9.98G [01:06<00:00, 178MB/s]
pytorch_model-00001-of-00002.bin:  99% 9.84G/9.98G [01:06<00:00, 181MB/s]
pytorch_model-00001-of-00002.bin:  99% 9.86G/9.98G [01:06<00:00, 166MB/s]
pytorch_model-00001-of-00002.bin:  99% 9.88G/9.98G [01:06<00:00, 167MB/s]
pytorch_model-00001-of-00002.bin:  99% 9.90G/9.98G [01:06<00:00, 171MB/s]
pytorch_model-00001-of-00002.bin:  99% 9.92G/9.98G [01:06<00:00, 176MB/s]
pytorch_model-00001-of-00002.bin: 100% 9.94G/9.98G [01:06<00:00, 179MB/s]
pytorch_model-00001-of-00002.bin: 100% 9.98G/9.98G [01:07<00:00, 149MB/s]
Downloading shards:  50% 1/2 [01:07<01:07, 67.54s/it]
pytorch_model-00002-of-00002.bin:   0% 0.00/3.50G [00:00<?, ?B/s]
pytorch_model-00002-of-00002.bin:   0% 10.5M/3.50G [00:00<01:15, 46.2MB/s]
pytorch_model-00002-of-00002.bin:   1% 21.0M/3.50G [00:00<00:52, 66.5MB/s]
pytorch_model-00002-of-00002.bin:   1% 41.9M/3.50G [00:00<00:36, 93.9MB/s]
pytorch_model-00002-of-00002.bin:   2% 62.9M/3.50G [00:00<00:29, 116MB/s] 
pytorch_model-00002-of-00002.bin:   2% 83.9M/3.50G [00:00<00:25, 135MB/s]
pytorch_model-00002-of-00002.bin:   3% 105M/3.50G [00:00<00:22, 149MB/s] 
pytorch_model-00002-of-00002.bin:   4% 126M/3.50G [00:00<00:20, 162MB/s]
pytorch_model-00002-of-00002.bin:   4% 147M/3.50G [00:01<00:19, 170MB/s]
pytorch_model-00002-of-00002.bin:   5% 168M/3.50G [00:01<00:18, 177MB/s]
pytorch_model-00002-of-00002.bin:   5% 189M/3.50G [00:01<00:18, 181MB/s]
pytorch_model-00002-of-00002.bin:   6% 210M/3.50G [00:01<00:17, 184MB/s]
pytorch_model-00002-of-00002.bin:   7% 231M/3.50G [00:01<00:17, 186MB/s]
pytorch_model-00002-of-00002.bin:   7% 252M/3.50G [00:01<00:17, 188MB/s]
pytorch_model-00002-of-00002.bin:   8% 273M/3.50G [00:01<00:17, 185MB/s]
pytorch_model-00002-of-00002.bin:   8% 294M/3.50G [00:01<00:17, 188MB/s]
pytorch_model-00002-of-00002.bin:   9% 315M/3.50G [00:01<00:16, 189MB/s]
pytorch_model-00002-of-00002.bin:  10% 336M/3.50G [00:02<00:16, 189MB/s]
pytorch_model-00002-of-00002.bin:  10% 357M/3.50G [00:02<00:16, 190MB/s]
pytorch_model-00002-of-00002.bin:  11% 377M/3.50G [00:02<00:16, 190MB/s]
pytorch_model-00002-of-00002.bin:  11% 398M/3.50G [00:02<00:16, 189MB/s]
pytorch_model-00002-of-00002.bin:  12% 419M/3.50G [00:02<00:16, 191MB/s]
pytorch_model-00002-of-00002.bin:  13% 440M/3.50G [00:02<00:16, 191MB/s]
pytorch_model-00002-of-00002.bin:  13% 461M/3.50G [00:02<00:15, 190MB/s]
pytorch_model-00002-of-00002.bin:  14% 482M/3.50G [00:02<00:15, 190MB/s]
pytorch_model-00002-of-00002.bin:  14% 503M/3.50G [00:02<00:15, 190MB/s]
pytorch_model-00002-of-00002.bin:  15% 524M/3.50G [00:03<00:15, 188MB/s]
pytorch_model-00002-of-00002.bin:  16% 545M/3.50G [00:03<00:16, 184MB/s]
pytorch_model-00002-of-00002.bin:  16% 566M/3.50G [00:03<00:15, 186MB/s]
pytorch_model-00002-of-00002.bin:  17% 587M/3.50G [00:03<00:15, 187MB/s]
pytorch_model-00002-of-00002.bin:  17% 608M/3.50G [00:03<00:15, 187MB/s]
pytorch_model-00002-of-00002.bin:  18% 629M/3.50G [00:03<00:15, 188MB/s]
pytorch_model-00002-of-00002.bin:  19% 650M/3.50G [00:03<00:15, 189MB/s]
pytorch_model-00002-of-00002.bin:  19% 671M/3.50G [00:03<00:15, 178MB/s]
pytorch_model-00002-of-00002.bin:  20% 692M/3.50G [00:03<00:15, 181MB/s]
pytorch_model-00002-of-00002.bin:  20% 713M/3.50G [00:04<00:15, 184MB/s]
pytorch_model-00002-of-00002.bin:  21% 734M/3.50G [00:04<00:14, 186MB/s]
pytorch_model-00002-of-00002.bin:  22% 755M/3.50G [00:04<00:14, 184MB/s]
pytorch_model-00002-of-00002.bin:  22% 776M/3.50G [00:04<00:14, 186MB/s]
pytorch_model-00002-of-00002.bin:  23% 797M/3.50G [00:04<00:14, 184MB/s]
pytorch_model-00002-of-00002.bin:  23% 818M/3.50G [00:04<00:14, 186MB/s]
pytorch_model-00002-of-00002.bin:  24% 839M/3.50G [00:04<00:14, 187MB/s]
pytorch_model-00002-of-00002.bin:  25% 860M/3.50G [00:04<00:14, 188MB/s]
pytorch_model-00002-of-00002.bin:  25% 881M/3.50G [00:04<00:13, 189MB/s]
pytorch_model-00002-of-00002.bin:  26% 902M/3.50G [00:05<00:13, 190MB/s]
pytorch_model-00002-of-00002.bin:  26% 923M/3.50G [00:05<00:13, 191MB/s]
pytorch_model-00002-of-00002.bin:  27% 944M/3.50G [00:05<00:13, 191MB/s]
pytorch_model-00002-of-00002.bin:  28% 965M/3.50G [00:05<00:13, 190MB/s]
pytorch_model-00002-of-00002.bin:  28% 986M/3.50G [00:05<00:13, 191MB/s]
pytorch_model-00002-of-00002.bin:  29% 1.01G/3.50G [00:05<00:13, 188MB/s]
pytorch_model-00002-of-00002.bin:  29% 1.03G/3.50G [00:05<00:13, 190MB/s]
pytorch_model-00002-of-00002.bin:  30% 1.05G/3.50G [00:05<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  31% 1.07G/3.50G [00:05<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  31% 1.09G/3.50G [00:06<00:12, 191MB/s]
pytorch_model-00002-of-00002.bin:  32% 1.11G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  32% 1.13G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  33% 1.15G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  34% 1.17G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  34% 1.20G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  35% 1.22G/3.50G [00:06<00:12, 190MB/s]
pytorch_model-00002-of-00002.bin:  35% 1.24G/3.50G [00:06<00:12, 188MB/s]
pytorch_model-00002-of-00002.bin:  36% 1.26G/3.50G [00:06<00:11, 190MB/s]
pytorch_model-00002-of-00002.bin:  37% 1.28G/3.50G [00:07<00:11, 189MB/s]
pytorch_model-00002-of-00002.bin:  37% 1.30G/3.50G [00:07<00:11, 190MB/s]
pytorch_model-00002-of-00002.bin:  38% 1.32G/3.50G [00:07<00:11, 189MB/s]
pytorch_model-00002-of-00002.bin:  38% 1.34G/3.50G [00:07<00:11, 190MB/s]
pytorch_model-00002-of-00002.bin:  39% 1.36G/3.50G [00:07<00:11, 190MB/s]
pytorch_model-00002-of-00002.bin:  40% 1.38G/3.50G [00:07<00:11, 191MB/s]
pytorch_model-00002-of-00002.bin:  40% 1.41G/3.50G [00:07<00:10, 191MB/s]
pytorch_model-00002-of-00002.bin:  41% 1.43G/3.50G [00:07<00:10, 190MB/s]
pytorch_model-00002-of-00002.bin:  41% 1.45G/3.50G [00:07<00:10, 190MB/s]
pytorch_model-00002-of-00002.bin:  42% 1.47G/3.50G [00:08<00:10, 191MB/s]
pytorch_model-00002-of-00002.bin:  43% 1.49G/3.50G [00:08<00:10, 191MB/s]
pytorch_model-00002-of-00002.bin:  43% 1.51G/3.50G [00:08<00:10, 189MB/s]
pytorch_model-00002-of-00002.bin:  44% 1.53G/3.50G [00:08<00:10, 188MB/s]
pytorch_model-00002-of-00002.bin:  44% 1.55G/3.50G [00:08<00:10, 191MB/s]
pytorch_model-00002-of-00002.bin:  45% 1.57G/3.50G [00:08<00:10, 190MB/s]
pytorch_model-00002-of-00002.bin:  46% 1.59G/3.50G [00:08<00:10, 190MB/s]
pytorch_model-00002-of-00002.bin:  46% 1.61G/3.50G [00:08<00:10, 188MB/s]
pytorch_model-00002-of-00002.bin:  47% 1.64G/3.50G [00:08<00:09, 187MB/s]
pytorch_model-00002-of-00002.bin:  47% 1.66G/3.50G [00:09<00:09, 189MB/s]
pytorch_model-00002-of-00002.bin:  48% 1.68G/3.50G [00:09<00:09, 190MB/s]
pytorch_model-00002-of-00002.bin:  49% 1.70G/3.50G [00:09<00:09, 190MB/s]
pytorch_model-00002-of-00002.bin:  49% 1.72G/3.50G [00:09<00:09, 190MB/s]
pytorch_model-00002-of-00002.bin:  50% 1.74G/3.50G [00:09<00:09, 191MB/s]
pytorch_model-00002-of-00002.bin:  50% 1.76G/3.50G [00:09<00:09, 189MB/s]
pytorch_model-00002-of-00002.bin:  51% 1.78G/3.50G [00:09<00:09, 190MB/s]
pytorch_model-00002-of-00002.bin:  52% 1.80G/3.50G [00:09<00:08, 190MB/s]
pytorch_model-00002-of-00002.bin:  52% 1.82G/3.50G [00:09<00:08, 190MB/s]
pytorch_model-00002-of-00002.bin:  53% 1.85G/3.50G [00:10<00:08, 185MB/s]
pytorch_model-00002-of-00002.bin:  53% 1.87G/3.50G [00:10<00:12, 132MB/s]
pytorch_model-00002-of-00002.bin:  54% 1.89G/3.50G [00:10<00:14, 110MB/s]
pytorch_model-00002-of-00002.bin:  55% 1.91G/3.50G [00:10<00:16, 98.4MB/s]
pytorch_model-00002-of-00002.bin:  55% 1.93G/3.50G [00:11<00:17, 89.5MB/s]
pytorch_model-00002-of-00002.bin:  55% 1.94G/3.50G [00:11<00:18, 86.0MB/s]
pytorch_model-00002-of-00002.bin:  56% 1.95G/3.50G [00:11<00:18, 84.6MB/s]
pytorch_model-00002-of-00002.bin:  56% 1.97G/3.50G [00:11<00:17, 89.0MB/s]
pytorch_model-00002-of-00002.bin:  57% 1.98G/3.50G [00:11<00:17, 86.5MB/s]
pytorch_model-00002-of-00002.bin:  57% 1.99G/3.50G [00:11<00:17, 84.4MB/s]
pytorch_model-00002-of-00002.bin:  57% 2.00G/3.50G [00:12<00:18, 81.8MB/s]
pytorch_model-00002-of-00002.bin:  58% 2.01G/3.50G [00:12<00:18, 81.6MB/s]
pytorch_model-00002-of-00002.bin:  58% 2.02G/3.50G [00:12<00:18, 79.0MB/s]
pytorch_model-00002-of-00002.bin:  58% 2.03G/3.50G [00:12<00:19, 76.7MB/s]
pytorch_model-00002-of-00002.bin:  59% 2.06G/3.50G [00:12<00:17, 80.5MB/s]
pytorch_model-00002-of-00002.bin:  59% 2.08G/3.50G [00:12<00:14, 102MB/s] 
pytorch_model-00002-of-00002.bin:  60% 2.10G/3.50G [00:12<00:11, 118MB/s]
pytorch_model-00002-of-00002.bin:  61% 2.12G/3.50G [00:13<00:10, 134MB/s]
pytorch_model-00002-of-00002.bin:  61% 2.14G/3.50G [00:13<00:09, 148MB/s]
pytorch_model-00002-of-00002.bin:  62% 2.16G/3.50G [00:13<00:12, 111MB/s]
pytorch_model-00002-of-00002.bin:  62% 2.18G/3.50G [00:13<00:13, 95.8MB/s]
pytorch_model-00002-of-00002.bin:  63% 2.20G/3.50G [00:13<00:13, 96.6MB/s]
pytorch_model-00002-of-00002.bin:  64% 2.22G/3.50G [00:14<00:14, 86.2MB/s]
pytorch_model-00002-of-00002.bin:  64% 2.23G/3.50G [00:14<00:14, 84.9MB/s]
pytorch_model-00002-of-00002.bin:  64% 2.24G/3.50G [00:14<00:15, 79.5MB/s]
pytorch_model-00002-of-00002.bin:  65% 2.26G/3.50G [00:14<00:12, 99.5MB/s]
pytorch_model-00002-of-00002.bin:  65% 2.29G/3.50G [00:14<00:12, 99.8MB/s]
pytorch_model-00002-of-00002.bin:  66% 2.31G/3.50G [00:15<00:10, 117MB/s] 
pytorch_model-00002-of-00002.bin:  67% 2.33G/3.50G [00:15<00:08, 131MB/s]
pytorch_model-00002-of-00002.bin:  67% 2.35G/3.50G [00:15<00:07, 146MB/s]
pytorch_model-00002-of-00002.bin:  68% 2.37G/3.50G [00:15<00:07, 155MB/s]
pytorch_model-00002-of-00002.bin:  68% 2.39G/3.50G [00:15<00:06, 164MB/s]
pytorch_model-00002-of-00002.bin:  69% 2.41G/3.50G [00:15<00:06, 171MB/s]
pytorch_model-00002-of-00002.bin:  69% 2.43G/3.50G [00:15<00:06, 177MB/s]
pytorch_model-00002-of-00002.bin:  70% 2.45G/3.50G [00:15<00:05, 181MB/s]
pytorch_model-00002-of-00002.bin:  71% 2.47G/3.50G [00:15<00:05, 184MB/s]
pytorch_model-00002-of-00002.bin:  71% 2.50G/3.50G [00:16<00:05, 185MB/s]
pytorch_model-00002-of-00002.bin:  72% 2.52G/3.50G [00:16<00:05, 187MB/s]
pytorch_model-00002-of-00002.bin:  72% 2.54G/3.50G [00:16<00:05, 188MB/s]
pytorch_model-00002-of-00002.bin:  73% 2.56G/3.50G [00:16<00:04, 189MB/s]
pytorch_model-00002-of-00002.bin:  74% 2.58G/3.50G [00:16<00:05, 183MB/s]
pytorch_model-00002-of-00002.bin:  74% 2.60G/3.50G [00:16<00:04, 182MB/s]
pytorch_model-00002-of-00002.bin:  75% 2.62G/3.50G [00:16<00:04, 184MB/s]
pytorch_model-00002-of-00002.bin:  75% 2.64G/3.50G [00:16<00:04, 184MB/s]
pytorch_model-00002-of-00002.bin:  76% 2.66G/3.50G [00:16<00:04, 185MB/s]
pytorch_model-00002-of-00002.bin:  77% 2.68G/3.50G [00:17<00:04, 186MB/s]
pytorch_model-00002-of-00002.bin:  77% 2.71G/3.50G [00:17<00:04, 188MB/s]
pytorch_model-00002-of-00002.bin:  78% 2.73G/3.50G [00:17<00:04, 189MB/s]
pytorch_model-00002-of-00002.bin:  78% 2.75G/3.50G [00:17<00:04, 187MB/s]
pytorch_model-00002-of-00002.bin:  79% 2.77G/3.50G [00:17<00:03, 189MB/s]
pytorch_model-00002-of-00002.bin:  80% 2.79G/3.50G [00:17<00:03, 189MB/s]
pytorch_model-00002-of-00002.bin:  80% 2.81G/3.50G [00:17<00:03, 190MB/s]
pytorch_model-00002-of-00002.bin:  81% 2.83G/3.50G [00:17<00:03, 190MB/s]
pytorch_model-00002-of-00002.bin:  81% 2.85G/3.50G [00:17<00:03, 189MB/s]
pytorch_model-00002-of-00002.bin:  82% 2.87G/3.50G [00:18<00:03, 191MB/s]
pytorch_model-00002-of-00002.bin:  83% 2.89G/3.50G [00:18<00:03, 189MB/s]
pytorch_model-00002-of-00002.bin:  83% 2.92G/3.50G [00:18<00:04, 125MB/s]
pytorch_model-00002-of-00002.bin:  84% 2.95G/3.50G [00:18<00:03, 158MB/s]
pytorch_model-00002-of-00002.bin:  85% 2.97G/3.50G [00:18<00:03, 166MB/s]
pytorch_model-00002-of-00002.bin:  85% 2.99G/3.50G [00:18<00:02, 171MB/s]
pytorch_model-00002-of-00002.bin:  86% 3.01G/3.50G [00:18<00:02, 176MB/s]
pytorch_model-00002-of-00002.bin:  87% 3.03G/3.50G [00:19<00:02, 180MB/s]
pytorch_model-00002-of-00002.bin:  87% 3.05G/3.50G [00:19<00:02, 184MB/s]
pytorch_model-00002-of-00002.bin:  88% 3.07G/3.50G [00:19<00:02, 185MB/s]
pytorch_model-00002-of-00002.bin:  88% 3.09G/3.50G [00:19<00:02, 185MB/s]
pytorch_model-00002-of-00002.bin:  89% 3.11G/3.50G [00:19<00:02, 183MB/s]
pytorch_model-00002-of-00002.bin:  90% 3.14G/3.50G [00:19<00:01, 187MB/s]
pytorch_model-00002-of-00002.bin:  90% 3.16G/3.50G [00:19<00:01, 188MB/s]
pytorch_model-00002-of-00002.bin:  91% 3.18G/3.50G [00:19<00:01, 189MB/s]
pytorch_model-00002-of-00002.bin:  91% 3.20G/3.50G [00:19<00:01, 190MB/s]
pytorch_model-00002-of-00002.bin:  92% 3.22G/3.50G [00:20<00:01, 189MB/s]
pytorch_model-00002-of-00002.bin:  93% 3.24G/3.50G [00:20<00:01, 190MB/s]
pytorch_model-00002-of-00002.bin:  93% 3.26G/3.50G [00:20<00:01, 161MB/s]
pytorch_model-00002-of-00002.bin:  94% 3.28G/3.50G [00:20<00:01, 168MB/s]
pytorch_model-00002-of-00002.bin:  94% 3.30G/3.50G [00:20<00:01, 175MB/s]
pytorch_model-00002-of-00002.bin:  95% 3.32G/3.50G [00:20<00:00, 180MB/s]
pytorch_model-00002-of-00002.bin:  96% 3.34G/3.50G [00:20<00:00, 182MB/s]
pytorch_model-00002-of-00002.bin:  96% 3.37G/3.50G [00:20<00:00, 184MB/s]
pytorch_model-00002-of-00002.bin:  97% 3.39G/3.50G [00:20<00:00, 186MB/s]
pytorch_model-00002-of-00002.bin:  97% 3.41G/3.50G [00:21<00:00, 188MB/s]
pytorch_model-00002-of-00002.bin:  98% 3.43G/3.50G [00:21<00:00, 188MB/s]
pytorch_model-00002-of-00002.bin:  99% 3.45G/3.50G [00:21<00:00, 188MB/s]
pytorch_model-00002-of-00002.bin:  99% 3.47G/3.50G [00:21<00:00, 189MB/s]
pytorch_model-00002-of-00002.bin: 100% 3.50G/3.50G [00:21<00:00, 162MB/s]
Downloading shards: 100% 2/2 [01:29<00:00, 44.63s/it]
[INFO|modeling_utils.py:1426] 2024-02-09 14:37:56,514 >> Instantiating LlamaForCausalLM model under default dtype torch.bfloat16.
[INFO|configuration_utils.py:826] 2024-02-09 14:37:56,516 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}

[INFO|modeling_utils.py:3615] 2024-02-09 14:37:58,602 >> Detected 4-bit loading: activating 4-bit loading for this model
Loading checkpoint shards:   0% 0/2 [00:00<?, ?it/s]/usr/local/lib/python3.10/dist-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100% 2/2 [00:09<00:00,  4.84s/it]
[INFO|modeling_utils.py:4350] 2024-02-09 14:38:08,482 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4358] 2024-02-09 14:38:08,482 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at elyza/ELYZA-japanese-Llama-2-7b-instruct.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
generation_config.json: 100% 154/154 [00:00<00:00, 1.03MB/s]
[INFO|configuration_utils.py:781] 2024-02-09 14:38:08,669 >> loading configuration file generation_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/generation_config.json
[INFO|configuration_utils.py:826] 2024-02-09 14:38:08,669 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "use_cache": false
}

02/09/2024 14:38:08 - INFO - llmtuner.model.patcher - Gradient checkpointing enabled.
02/09/2024 14:38:08 - INFO - llmtuner.model.adapter - Fine-tuning method: LoRA
02/09/2024 14:38:09 - INFO - llmtuner.model.loader - trainable params: 4194304 || all params: 6742609920 || trainable%: 0.0622
https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon/resolve/c9da5340ae4343592fd50e5445ed16e51b7600b3/README.md not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/0e6d90c7845bc4dfc6334143d5c19d24849f9a15c956129b60195ba667f51c64.15093dd0724c13fd2b96c4ec76aecd2b86fa4bc56fdc3546abeced21f8fe843a.incomplete
Downloading readme: 100% 296/296 [00:00<00:00, 3.08MB/s]
storing https://huggingface.co/datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon/resolve/c9da5340ae4343592fd50e5445ed16e51b7600b3/README.md in cache at /root/.cache/huggingface/datasets/downloads/0e6d90c7845bc4dfc6334143d5c19d24849f9a15c956129b60195ba667f51c64.15093dd0724c13fd2b96c4ec76aecd2b86fa4bc56fdc3546abeced21f8fe843a
creating metadata file for /root/.cache/huggingface/datasets/downloads/0e6d90c7845bc4dfc6334143d5c19d24849f9a15c956129b60195ba667f51c64.15093dd0724c13fd2b96c4ec76aecd2b86fa4bc56fdc3546abeced21f8fe843a
No config specified, defaulting to the single config: databricks-dolly-15k-ja-gozarinnemon/default
Loading Dataset Infos from /usr/local/lib/python3.10/dist-packages/datasets/packaged_modules/json
Generating dataset databricks-dolly-15k-ja-gozarinnemon (/root/.cache/huggingface/datasets/bbz662bbz___databricks-dolly-15k-ja-gozarinnemon/default/0.0.0/c9da5340ae4343592fd50e5445ed16e51b7600b3)
Downloading and preparing dataset databricks-dolly-15k-ja-gozarinnemon/default to /root/.cache/huggingface/datasets/bbz662bbz___databricks-dolly-15k-ja-gozarinnemon/default/0.0.0/c9da5340ae4343592fd50e5445ed16e51b7600b3...
Dataset not on Hf google storage. Downloading and preparing it from source
hf://datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon@c9da5340ae4343592fd50e5445ed16e51b7600b3/databricks-dolly-15k-ja-gozarinnemon.json not found in cache or force_download set to True, downloading to /root/.cache/huggingface/datasets/downloads/e1ff4fa7026dcb49ea28d332fedc43da14149b80ba354522f6e08525025c01c9.incomplete
Downloading data: 100% 18.2M/18.2M [00:01<00:00, 11.7MB/s]
storing hf://datasets/bbz662bbz/databricks-dolly-15k-ja-gozarinnemon@c9da5340ae4343592fd50e5445ed16e51b7600b3/databricks-dolly-15k-ja-gozarinnemon.json in cache at /root/.cache/huggingface/datasets/downloads/e1ff4fa7026dcb49ea28d332fedc43da14149b80ba354522f6e08525025c01c9
creating metadata file for /root/.cache/huggingface/datasets/downloads/e1ff4fa7026dcb49ea28d332fedc43da14149b80ba354522f6e08525025c01c9
Downloading took 0.0 min
Checksum Computation took 0.0 min
Generating train split
Generating train split: 15015 examples [00:00, 61731.61 examples/s]
Unable to verify splits sizes.
Dataset databricks-dolly-15k-ja-gozarinnemon downloaded and prepared to /root/.cache/huggingface/datasets/bbz662bbz___databricks-dolly-15k-ja-gozarinnemon/default/0.0.0/c9da5340ae4343592fd50e5445ed16e51b7600b3. Subsequent calls will reuse this data.
Converting format of dataset:   0% 0/10000 [00:00<?, ? examples/s]Caching processed dataset at /root/.cache/huggingface/datasets/bbz662bbz___databricks-dolly-15k-ja-gozarinnemon/default/0.0.0/c9da5340ae4343592fd50e5445ed16e51b7600b3/cache-dec56b9b6e657e66.arrow
Converting format of dataset: 100% 10000/10000 [00:00<00:00, 60993.77 examples/s]
Running tokenizer on dataset:   0% 0/10000 [00:00<?, ? examples/s]Caching processed dataset at /root/.cache/huggingface/datasets/bbz662bbz___databricks-dolly-15k-ja-gozarinnemon/default/0.0.0/c9da5340ae4343592fd50e5445ed16e51b7600b3/cache-ba1e011674100b9a.arrow
Running tokenizer on dataset: 100% 10000/10000 [00:19<00:00, 521.67 examples/s]
input_ids:
[1, 518, 25580, 29962, 3532, 14816, 29903, 6778, 13, 30641, 30371, 30366, 30449, 235, 173, 163, 31525, 30499, 232, 135, 173, 31701, 30371, 30325, 30346, 30313, 30199, 30310, 30373, 30255, 30369, 30203, 30279, 30499, 30427, 30267, 13, 29966, 829, 14816, 29903, 6778, 13, 13, 31270, 30897, 30185, 30391, 30203, 30290, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 31727, 30816, 30449, 30298, 30773, 30412, 30513, 236, 132, 142, 31727, 30396, 31404, 31020, 30326, 30366, 30199, 30499, 30427, 30412, 30882, 13, 31270, 30897, 30185, 30391, 30203, 30290, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 31727, 30816, 30419, 29963, 381, 5359, 8314, 29718, 349, 1017, 19806, 30409, 30449, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 30396, 233, 142, 163, 30940, 30364, 30427, 30332, 31270, 30897, 30185, 30391, 30203, 30290, 30582, 30281, 30203, 30335, 30396, 232, 137, 163, 30427, 30332, 30878, 30257, 30199, 235, 139, 188, 232, 158, 166, 235, 169, 146, 31382, 30396, 31695, 30773, 31727, 30816, 30437, 30564, 30499, 30427, 30267, 29906, 29900, 29900, 29900, 30470, 29947, 30534, 29941, 29896, 30325, 30353, 30330, 31270, 30897, 30185, 30391, 30203, 30290, 30582, 30258, 30185, 30816, 31449, 30364, 30326, 30466, 30330, 29906, 31540, 30199, 31727, 30816, 31540, 30330, 29896, 30773, 30199, 30816, 30874, 30396, 236, 132, 142, 30448, 30326, 30466, 30615, 30185, 30809, 30255, 30396, 31404, 31020, 30326, 30441, 30326, 30366, 30267, 29906, 29900, 29900, 29896, 30470, 29929, 30534, 30199, 30310, 30203, 30885, 30317, 30279, 30290, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 30816, 31449, 30199, 232, 183, 172, 232, 166, 141, 31220, 30330, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 30199, 30356, 30728, 30461, 31045, 30499, 233, 131, 168, 31859, 30353, 30533, 30956, 30396, 234, 165, 189, 30939, 30326, 30441, 30326, 30366, 30267, 31110, 30199, 31220, 30449, 30582, 30303, 30255, 31061, 30203, 30330, 30604, 30258, 31003, 30258, 30203, 30330, 30373, 30335, 30635, 30185, 30396, 30758, 30582, 30364, 30326, 30466, 30330, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 30356, 30728, 30199, 29941, 29906, 30769, 30461, 30353, 31157, 31092, 231, 188, 154, 30453, 30752, 30553, 30332, 30441, 30499, 30353, 30494, 30899, 30326, 30441, 30326, 30366, 30267, 518, 29914, 25580, 29962, 29871, 30672, 30330, 30453, 30389, 30914, 30723, 30389, 30449, 31579, 30465, 30267, 29871, 31270, 30897, 30185, 30391, 30203, 30290, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 31727, 30816, 30449, 30330, 29906, 29900, 29900, 29900, 30470, 29947, 30534, 29941, 29896, 30325, 30353, 31270, 30897, 30185, 30391, 30203, 30290, 30582, 30258, 30185, 31727, 30816, 30364, 30326, 30466, 30330, 29906, 31540, 30199, 31727, 30816, 31540, 30499, 232, 144, 155, 30287, 30874, 31357, 30199, 236, 132, 142, 31727, 30396, 31404, 31020, 30326, 30441, 30326, 30366, 30499, 31622, 230, 132, 153, 30332, 30267, 31043, 30513, 30389, 30807, 31250, 30267, 2]
inputs:
<s> [INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。
<</SYS>>

ヴァージン・オーストラリア航空はいつから運航を開始したのですか？
ヴァージン・オーストラリア航空（Virgin Australia Airlines Pty Ltd）はオーストラリアを拠点とするヴァージン・ブランドを冠する最大の船団規模を持つ航空会社です。2000年8月31日に、ヴァージン・ブルー空港として、2機の航空機、1つの空路を運行してサービスを開始しました。2001年9月のアンセット・オーストラリア空港の崩壊後、オーストラリアの国内市場で急速に地位を確立しました。その後はブリスベン、メルボルン、シドニーをハブとして、オーストラリア国内の32都市に直接乗り入れるまでに成長しました。 [/INST] 我、りんえもんは思う。 ヴァージン・オーストラリア航空は、2000年8月31日にヴァージン・ブルー航空として、2機の航空機で単一路線の運航を開始しましたでござる。知らんけど。</s>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 29871, 30672, 30330, 30453, 30389, 30914, 30723, 30389, 30449, 31579, 30465, 30267, 29871, 31270, 30897, 30185, 30391, 30203, 30290, 30514, 30185, 30255, 30279, 30281, 30303, 30310, 31727, 30816, 30449, 30330, 29906, 29900, 29900, 29900, 30470, 29947, 30534, 29941, 29896, 30325, 30353, 31270, 30897, 30185, 30391, 30203, 30290, 30582, 30258, 30185, 31727, 30816, 30364, 30326, 30466, 30330, 29906, 31540, 30199, 31727, 30816, 31540, 30499, 232, 144, 155, 30287, 30874, 31357, 30199, 236, 132, 142, 31727, 30396, 31404, 31020, 30326, 30441, 30326, 30366, 30499, 31622, 230, 132, 153, 30332, 30267, 31043, 30513, 30389, 30807, 31250, 30267, 2]
labels:
我、りんえもんは思う。 ヴァージン・オーストラリア航空は、2000年8月31日にヴァージン・ブルー航空として、2機の航空機で単一路線の運航を開始しましたでござる。知らんけど。</s>
[INFO|training_args.py:1828] 2024-02-09 14:38:33,199 >> PyTorch: setting up devices
[INFO|trainer.py:571] 2024-02-09 14:38:33,212 >> Using auto half precision backend
[INFO|trainer.py:1721] 2024-02-09 14:38:33,506 >> ***** Running training *****
[INFO|trainer.py:1722] 2024-02-09 14:38:33,506 >>   Num examples = 10,000
[INFO|trainer.py:1723] 2024-02-09 14:38:33,506 >>   Num Epochs = 3
[INFO|trainer.py:1724] 2024-02-09 14:38:33,506 >>   Instantaneous batch size per device = 16
[INFO|trainer.py:1727] 2024-02-09 14:38:33,506 >>   Total train batch size (w. parallel, distributed & accumulation) = 64
[INFO|trainer.py:1728] 2024-02-09 14:38:33,506 >>   Gradient Accumulation steps = 4
[INFO|trainer.py:1729] 2024-02-09 14:38:33,506 >>   Total optimization steps = 468
[INFO|trainer.py:1730] 2024-02-09 14:38:33,508 >>   Number of trainable parameters = 4,194,304
02/09/2024 14:39:26 - INFO - llmtuner.extras.callbacks - {'loss': 1.6971, 'learning_rate': 1.9994e-04, 'epoch': 0.03}
{'loss': 1.6971, 'learning_rate': 0.00019994367810210973, 'epoch': 0.03}
02/09/2024 14:40:18 - INFO - llmtuner.extras.callbacks - {'loss': 1.4355, 'learning_rate': 1.9977e-04, 'epoch': 0.06}
{'loss': 1.4355, 'learning_rate': 0.00019977477585156252, 'epoch': 0.06}
02/09/2024 14:41:10 - INFO - llmtuner.extras.callbacks - {'loss': 1.2430, 'learning_rate': 1.9949e-04, 'epoch': 0.10}
{'loss': 1.243, 'learning_rate': 0.00019949348350626456, 'epoch': 0.1}
02/09/2024 14:42:02 - INFO - llmtuner.extras.callbacks - {'loss': 1.0422, 'learning_rate': 1.9910e-04, 'epoch': 0.13}
{'loss': 1.0422, 'learning_rate': 0.00019910011792459087, 'epoch': 0.13}
02/09/2024 14:42:54 - INFO - llmtuner.extras.callbacks - {'loss': 0.9979, 'learning_rate': 1.9860e-04, 'epoch': 0.16}
{'loss': 0.9979, 'learning_rate': 0.00019859512220846387, 'epoch': 0.16}
02/09/2024 14:43:46 - INFO - llmtuner.extras.callbacks - {'loss': 0.9667, 'learning_rate': 1.9798e-04, 'epoch': 0.19}
{'loss': 0.9667, 'learning_rate': 0.00019797906520422677, 'epoch': 0.19}
02/09/2024 14:44:39 - INFO - llmtuner.extras.callbacks - {'loss': 0.9448, 'learning_rate': 1.9725e-04, 'epoch': 0.22}
{'loss': 0.9448, 'learning_rate': 0.00019725264086187334, 'epoch': 0.22}
02/09/2024 14:45:31 - INFO - llmtuner.extras.callbacks - {'loss': 0.8960, 'learning_rate': 1.9642e-04, 'epoch': 0.26}
{'loss': 0.896, 'learning_rate': 0.00019641666745335624, 'epoch': 0.26}
02/09/2024 14:46:23 - INFO - llmtuner.extras.callbacks - {'loss': 0.8622, 'learning_rate': 1.9547e-04, 'epoch': 0.29}
{'loss': 0.8622, 'learning_rate': 0.00019547208665085457, 'epoch': 0.29}
02/09/2024 14:47:15 - INFO - llmtuner.extras.callbacks - {'loss': 0.8280, 'learning_rate': 1.9442e-04, 'epoch': 0.32}
{'loss': 0.828, 'learning_rate': 0.00019441996246603846, 'epoch': 0.32}
02/09/2024 14:48:07 - INFO - llmtuner.extras.callbacks - {'loss': 0.8674, 'learning_rate': 1.9326e-04, 'epoch': 0.35}
{'loss': 0.8674, 'learning_rate': 0.00019326148005152606, 'epoch': 0.35}
02/09/2024 14:48:59 - INFO - llmtuner.extras.callbacks - {'loss': 0.8654, 'learning_rate': 1.9200e-04, 'epoch': 0.38}
{'loss': 0.8654, 'learning_rate': 0.00019199794436588243, 'epoch': 0.38}
02/09/2024 14:49:51 - INFO - llmtuner.extras.callbacks - {'loss': 0.8928, 'learning_rate': 1.9063e-04, 'epoch': 0.42}
{'loss': 0.8928, 'learning_rate': 0.000190630778703665, 'epoch': 0.42}
02/09/2024 14:50:43 - INFO - llmtuner.extras.callbacks - {'loss': 0.8897, 'learning_rate': 1.8916e-04, 'epoch': 0.45}
{'loss': 0.8897, 'learning_rate': 0.0001891615230921703, 'epoch': 0.45}
02/09/2024 14:51:35 - INFO - llmtuner.extras.callbacks - {'loss': 0.8326, 'learning_rate': 1.8759e-04, 'epoch': 0.48}
{'loss': 0.8326, 'learning_rate': 0.0001875918325566888, 'epoch': 0.48}
02/09/2024 14:52:27 - INFO - llmtuner.extras.callbacks - {'loss': 0.8833, 'learning_rate': 1.8592e-04, 'epoch': 0.51}
{'loss': 0.8833, 'learning_rate': 0.0001859234752562217, 'epoch': 0.51}
02/09/2024 14:53:19 - INFO - llmtuner.extras.callbacks - {'loss': 0.8819, 'learning_rate': 1.8416e-04, 'epoch': 0.54}
{'loss': 0.8819, 'learning_rate': 0.00018415833049175941, 'epoch': 0.54}
02/09/2024 14:54:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.8520, 'learning_rate': 1.8230e-04, 'epoch': 0.58}
{'loss': 0.852, 'learning_rate': 0.00018229838658936564, 'epoch': 0.58}
02/09/2024 14:55:03 - INFO - llmtuner.extras.callbacks - {'loss': 0.8665, 'learning_rate': 1.8035e-04, 'epoch': 0.61}
{'loss': 0.8665, 'learning_rate': 0.00018034573866045146, 'epoch': 0.61}
02/09/2024 14:55:56 - INFO - llmtuner.extras.callbacks - {'loss': 0.8454, 'learning_rate': 1.7830e-04, 'epoch': 0.64}
{'loss': 0.8454, 'learning_rate': 0.00017830258624176225, 'epoch': 0.64}
[INFO|trainer.py:2936] 2024-02-09 14:55:56,041 >> Saving model checkpoint to saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-100
[INFO|configuration_utils.py:729] 2024-02-09 14:55:56,340 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 14:55:56,341 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-09 14:55:56,386 >> tokenizer config file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-100/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-09 14:55:56,387 >> Special tokens file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-100/special_tokens_map.json
02/09/2024 14:56:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.8637, 'learning_rate': 1.7617e-04, 'epoch': 0.67}
{'loss': 0.8637, 'learning_rate': 0.00017617123081773591, 'epoch': 0.67}
02/09/2024 14:57:40 - INFO - llmtuner.extras.callbacks - {'loss': 0.8400, 'learning_rate': 1.7395e-04, 'epoch': 0.70}
{'loss': 0.84, 'learning_rate': 0.00017395407322802372, 'epoch': 0.7}
02/09/2024 14:58:32 - INFO - llmtuner.extras.callbacks - {'loss': 0.8424, 'learning_rate': 1.7165e-04, 'epoch': 0.74}
{'loss': 0.8424, 'learning_rate': 0.00017165361096309318, 'epoch': 0.74}
02/09/2024 14:59:24 - INFO - llmtuner.extras.callbacks - {'loss': 0.8334, 'learning_rate': 1.6927e-04, 'epoch': 0.77}
{'loss': 0.8334, 'learning_rate': 0.00016927243535095997, 'epoch': 0.77}
02/09/2024 15:00:16 - INFO - llmtuner.extras.callbacks - {'loss': 0.8694, 'learning_rate': 1.6681e-04, 'epoch': 0.80}
{'loss': 0.8694, 'learning_rate': 0.00016681322863821776, 'epoch': 0.8}
02/09/2024 15:01:09 - INFO - llmtuner.extras.callbacks - {'loss': 0.8479, 'learning_rate': 1.6428e-04, 'epoch': 0.83}
{'loss': 0.8479, 'learning_rate': 0.00016427876096865394, 'epoch': 0.83}
02/09/2024 15:02:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.8422, 'learning_rate': 1.6167e-04, 'epoch': 0.86}
{'loss': 0.8422, 'learning_rate': 0.00016167188726285434, 'epoch': 0.86}
02/09/2024 15:02:52 - INFO - llmtuner.extras.callbacks - {'loss': 0.8943, 'learning_rate': 1.5900e-04, 'epoch': 0.90}
{'loss': 0.8943, 'learning_rate': 0.00015899554400231232, 'epoch': 0.9}
02/09/2024 15:03:45 - INFO - llmtuner.extras.callbacks - {'loss': 0.8336, 'learning_rate': 1.5625e-04, 'epoch': 0.93}
{'loss': 0.8336, 'learning_rate': 0.00015625274592166467, 'epoch': 0.93}
02/09/2024 15:04:37 - INFO - llmtuner.extras.callbacks - {'loss': 0.8532, 'learning_rate': 1.5345e-04, 'epoch': 0.96}
{'loss': 0.8532, 'learning_rate': 0.0001534465826127801, 'epoch': 0.96}
02/09/2024 15:05:29 - INFO - llmtuner.extras.callbacks - {'loss': 0.8630, 'learning_rate': 1.5058e-04, 'epoch': 0.99}
{'loss': 0.863, 'learning_rate': 0.00015058021504452552, 'epoch': 0.99}
02/09/2024 15:06:21 - INFO - llmtuner.extras.callbacks - {'loss': 0.8080, 'learning_rate': 1.4766e-04, 'epoch': 1.02}
{'loss': 0.808, 'learning_rate': 0.0001476568720021308, 'epoch': 1.02}
02/09/2024 15:07:13 - INFO - llmtuner.extras.callbacks - {'loss': 0.8018, 'learning_rate': 1.4468e-04, 'epoch': 1.06}
{'loss': 0.8018, 'learning_rate': 0.00014467984645016258, 'epoch': 1.06}
02/09/2024 15:08:05 - INFO - llmtuner.extras.callbacks - {'loss': 0.8585, 'learning_rate': 1.4165e-04, 'epoch': 1.09}
{'loss': 0.8585, 'learning_rate': 0.00014165249182320402, 'epoch': 1.09}
02/09/2024 15:08:57 - INFO - llmtuner.extras.callbacks - {'loss': 0.8271, 'learning_rate': 1.3858e-04, 'epoch': 1.12}
{'loss': 0.8271, 'learning_rate': 0.00013857821824841854, 'epoch': 1.12}
02/09/2024 15:09:49 - INFO - llmtuner.extras.callbacks - {'loss': 0.7957, 'learning_rate': 1.3546e-04, 'epoch': 1.15}
{'loss': 0.7957, 'learning_rate': 0.00013546048870425356, 'epoch': 1.15}
02/09/2024 15:10:42 - INFO - llmtuner.extras.callbacks - {'loss': 0.8038, 'learning_rate': 1.3230e-04, 'epoch': 1.18}
{'loss': 0.8038, 'learning_rate': 0.0001323028151196098, 'epoch': 1.18}
02/09/2024 15:11:34 - INFO - llmtuner.extras.callbacks - {'loss': 0.7929, 'learning_rate': 1.2911e-04, 'epoch': 1.22}
{'loss': 0.7929, 'learning_rate': 0.00012910875441787128, 'epoch': 1.22}
02/09/2024 15:12:26 - INFO - llmtuner.extras.callbacks - {'loss': 0.8211, 'learning_rate': 1.2588e-04, 'epoch': 1.25}
{'loss': 0.8211, 'learning_rate': 0.00012588190451025207, 'epoch': 1.25}
02/09/2024 15:13:18 - INFO - llmtuner.extras.callbacks - {'loss': 0.8283, 'learning_rate': 1.2263e-04, 'epoch': 1.28}
{'loss': 0.8283, 'learning_rate': 0.00012262590024297225, 'epoch': 1.28}
[INFO|trainer.py:2936] 2024-02-09 15:13:18,514 >> Saving model checkpoint to saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-200
[INFO|configuration_utils.py:729] 2024-02-09 15:13:18,822 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 15:13:18,823 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-09 15:13:18,863 >> tokenizer config file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-200/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-09 15:13:18,863 >> Special tokens file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-200/special_tokens_map.json
02/09/2024 15:14:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.8396, 'learning_rate': 1.1934e-04, 'epoch': 1.31}
{'loss': 0.8396, 'learning_rate': 0.00011934440930282913, 'epoch': 1.31}
02/09/2024 15:15:03 - INFO - llmtuner.extras.callbacks - {'loss': 0.8111, 'learning_rate': 1.1604e-04, 'epoch': 1.34}
{'loss': 0.8111, 'learning_rate': 0.00011604112808577603, 'epoch': 1.34}
02/09/2024 15:15:55 - INFO - llmtuner.extras.callbacks - {'loss': 0.8193, 'learning_rate': 1.1272e-04, 'epoch': 1.38}
{'loss': 0.8193, 'learning_rate': 0.0001127197775331611, 'epoch': 1.38}
02/09/2024 15:16:47 - INFO - llmtuner.extras.callbacks - {'loss': 0.8449, 'learning_rate': 1.0938e-04, 'epoch': 1.41}
{'loss': 0.8449, 'learning_rate': 0.00010938409894031794, 'epoch': 1.41}
02/09/2024 15:17:39 - INFO - llmtuner.extras.callbacks - {'loss': 0.7992, 'learning_rate': 1.0604e-04, 'epoch': 1.44}
{'loss': 0.7992, 'learning_rate': 0.00010603784974222861, 'epoch': 1.44}
02/09/2024 15:18:31 - INFO - llmtuner.extras.callbacks - {'loss': 0.8035, 'learning_rate': 1.0268e-04, 'epoch': 1.47}
{'loss': 0.8035, 'learning_rate': 0.00010268479928100614, 'epoch': 1.47}
02/09/2024 15:19:23 - INFO - llmtuner.extras.callbacks - {'loss': 0.8268, 'learning_rate': 9.9329e-05, 'epoch': 1.50}
{'loss': 0.8268, 'learning_rate': 9.93287245599644e-05, 'epoch': 1.5}
02/09/2024 15:20:15 - INFO - llmtuner.extras.callbacks - {'loss': 0.8310, 'learning_rate': 9.5973e-05, 'epoch': 1.54}
{'loss': 0.831, 'learning_rate': 9.597340598905852e-05, 'epoch': 1.54}
02/09/2024 15:21:08 - INFO - llmtuner.extras.callbacks - {'loss': 0.8032, 'learning_rate': 9.2623e-05, 'epoch': 1.57}
{'loss': 0.8032, 'learning_rate': 9.262262312648717e-05, 'epoch': 1.57}
02/09/2024 15:22:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.8159, 'learning_rate': 8.9280e-05, 'epoch': 1.60}
{'loss': 0.8159, 'learning_rate': 8.928015042125523e-05, 'epoch': 1.6}
02/09/2024 15:22:52 - INFO - llmtuner.extras.callbacks - {'loss': 0.7979, 'learning_rate': 8.5950e-05, 'epoch': 1.63}
{'loss': 0.7979, 'learning_rate': 8.594975296149076e-05, 'epoch': 1.63}
02/09/2024 15:23:44 - INFO - llmtuner.extras.callbacks - {'loss': 0.7869, 'learning_rate': 8.2635e-05, 'epoch': 1.66}
{'loss': 0.7869, 'learning_rate': 8.263518223330697e-05, 'epoch': 1.66}
02/09/2024 15:24:36 - INFO - llmtuner.extras.callbacks - {'loss': 0.7794, 'learning_rate': 7.9340e-05, 'epoch': 1.70}
{'loss': 0.7794, 'learning_rate': 7.9340171894986e-05, 'epoch': 1.7}
02/09/2024 15:25:28 - INFO - llmtuner.extras.callbacks - {'loss': 0.7798, 'learning_rate': 7.6068e-05, 'epoch': 1.73}
{'loss': 0.7798, 'learning_rate': 7.606843357124426e-05, 'epoch': 1.73}
02/09/2024 15:26:19 - INFO - llmtuner.extras.callbacks - {'loss': 0.8084, 'learning_rate': 7.2824e-05, 'epoch': 1.76}
{'loss': 0.8084, 'learning_rate': 7.282365267231756e-05, 'epoch': 1.76}
02/09/2024 15:27:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.8141, 'learning_rate': 6.9609e-05, 'epoch': 1.79}
{'loss': 0.8141, 'learning_rate': 6.960948424257532e-05, 'epoch': 1.79}
02/09/2024 15:28:04 - INFO - llmtuner.extras.callbacks - {'loss': 0.8010, 'learning_rate': 6.6430e-05, 'epoch': 1.82}
{'loss': 0.801, 'learning_rate': 6.642954884333955e-05, 'epoch': 1.82}
02/09/2024 15:28:56 - INFO - llmtuner.extras.callbacks - {'loss': 0.8173, 'learning_rate': 6.3287e-05, 'epoch': 1.86}
{'loss': 0.8173, 'learning_rate': 6.328742847454724e-05, 'epoch': 1.86}
02/09/2024 15:29:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.8816, 'learning_rate': 6.0187e-05, 'epoch': 1.89}
{'loss': 0.8816, 'learning_rate': 6.01866625398499e-05, 'epoch': 1.89}
02/09/2024 15:30:40 - INFO - llmtuner.extras.callbacks - {'loss': 0.8000, 'learning_rate': 5.7131e-05, 'epoch': 1.92}
{'loss': 0.8, 'learning_rate': 5.713074385969457e-05, 'epoch': 1.92}
[INFO|trainer.py:2936] 2024-02-09 15:30:40,514 >> Saving model checkpoint to saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-300
[INFO|configuration_utils.py:729] 2024-02-09 15:30:40,812 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 15:30:40,813 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-09 15:30:40,854 >> tokenizer config file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-300/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-09 15:30:40,855 >> Special tokens file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-300/special_tokens_map.json
02/09/2024 15:31:33 - INFO - llmtuner.extras.callbacks - {'loss': 0.8039, 'learning_rate': 5.4123e-05, 'epoch': 1.95}
{'loss': 0.8039, 'learning_rate': 5.412311473687859e-05, 'epoch': 1.95}
02/09/2024 15:32:25 - INFO - llmtuner.extras.callbacks - {'loss': 0.8172, 'learning_rate': 5.1167e-05, 'epoch': 1.98}
{'loss': 0.8172, 'learning_rate': 5.116716307900893e-05, 'epoch': 1.98}
02/09/2024 15:33:17 - INFO - llmtuner.extras.callbacks - {'loss': 0.8272, 'learning_rate': 4.8266e-05, 'epoch': 2.02}
{'loss': 0.8272, 'learning_rate': 4.826621858223431e-05, 'epoch': 2.02}
02/09/2024 15:34:09 - INFO - llmtuner.extras.callbacks - {'loss': 0.7993, 'learning_rate': 4.5424e-05, 'epoch': 2.05}
{'loss': 0.7993, 'learning_rate': 4.542354898054953e-05, 'epoch': 2.05}
02/09/2024 15:35:01 - INFO - llmtuner.extras.callbacks - {'loss': 0.8234, 'learning_rate': 4.2642e-05, 'epoch': 2.08}
{'loss': 0.8234, 'learning_rate': 4.264235636489542e-05, 'epoch': 2.08}
02/09/2024 15:35:53 - INFO - llmtuner.extras.callbacks - {'loss': 0.8056, 'learning_rate': 3.9926e-05, 'epoch': 2.11}
{'loss': 0.8056, 'learning_rate': 3.99257735762021e-05, 'epoch': 2.11}
02/09/2024 15:36:45 - INFO - llmtuner.extras.callbacks - {'loss': 0.8109, 'learning_rate': 3.7277e-05, 'epoch': 2.14}
{'loss': 0.8109, 'learning_rate': 3.72768606764384e-05, 'epoch': 2.14}
02/09/2024 15:37:38 - INFO - llmtuner.extras.callbacks - {'loss': 0.7896, 'learning_rate': 3.4699e-05, 'epoch': 2.18}
{'loss': 0.7896, 'learning_rate': 3.469860150164152e-05, 'epoch': 2.18}
02/09/2024 15:38:30 - INFO - llmtuner.extras.callbacks - {'loss': 0.7981, 'learning_rate': 3.2194e-05, 'epoch': 2.21}
{'loss': 0.7981, 'learning_rate': 3.219390030081091e-05, 'epoch': 2.21}
02/09/2024 15:39:22 - INFO - llmtuner.extras.callbacks - {'loss': 0.8136, 'learning_rate': 2.9766e-05, 'epoch': 2.24}
{'loss': 0.8136, 'learning_rate': 2.976557846445225e-05, 'epoch': 2.24}
02/09/2024 15:40:14 - INFO - llmtuner.extras.callbacks - {'loss': 0.8211, 'learning_rate': 2.7416e-05, 'epoch': 2.27}
{'loss': 0.8211, 'learning_rate': 2.7416371346455792e-05, 'epoch': 2.27}
02/09/2024 15:41:06 - INFO - llmtuner.extras.callbacks - {'loss': 0.7787, 'learning_rate': 2.5149e-05, 'epoch': 2.30}
{'loss': 0.7787, 'learning_rate': 2.514892518288988e-05, 'epoch': 2.3}
02/09/2024 15:41:58 - INFO - llmtuner.extras.callbacks - {'loss': 0.7686, 'learning_rate': 2.2966e-05, 'epoch': 2.34}
{'loss': 0.7686, 'learning_rate': 2.296579411118055e-05, 'epoch': 2.34}
02/09/2024 15:42:50 - INFO - llmtuner.extras.callbacks - {'loss': 0.7938, 'learning_rate': 2.0869e-05, 'epoch': 2.37}
{'loss': 0.7938, 'learning_rate': 2.0869437293033835e-05, 'epoch': 2.37}
02/09/2024 15:43:42 - INFO - llmtuner.extras.callbacks - {'loss': 0.8037, 'learning_rate': 1.8862e-05, 'epoch': 2.40}
{'loss': 0.8037, 'learning_rate': 1.8862216144342692e-05, 'epoch': 2.4}
02/09/2024 15:44:34 - INFO - llmtuner.extras.callbacks - {'loss': 0.8047, 'learning_rate': 1.6946e-05, 'epoch': 2.43}
{'loss': 0.8047, 'learning_rate': 1.6946391675198836e-05, 'epoch': 2.43}
02/09/2024 15:45:27 - INFO - llmtuner.extras.callbacks - {'loss': 0.7960, 'learning_rate': 1.5124e-05, 'epoch': 2.46}
{'loss': 0.796, 'learning_rate': 1.5124121943004766e-05, 'epoch': 2.46}
02/09/2024 15:46:19 - INFO - llmtuner.extras.callbacks - {'loss': 0.7619, 'learning_rate': 1.3397e-05, 'epoch': 2.50}
{'loss': 0.7619, 'learning_rate': 1.339745962155613e-05, 'epoch': 2.5}
02/09/2024 15:47:11 - INFO - llmtuner.extras.callbacks - {'loss': 0.8000, 'learning_rate': 1.1768e-05, 'epoch': 2.53}
{'loss': 0.8, 'learning_rate': 1.1768349688832203e-05, 'epoch': 2.53}
02/09/2024 15:48:03 - INFO - llmtuner.extras.callbacks - {'loss': 0.7904, 'learning_rate': 1.0239e-05, 'epoch': 2.56}
{'loss': 0.7904, 'learning_rate': 1.0238627236098619e-05, 'epoch': 2.56}
[INFO|trainer.py:2936] 2024-02-09 15:48:03,456 >> Saving model checkpoint to saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-400
[INFO|configuration_utils.py:729] 2024-02-09 15:48:03,971 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 15:48:03,972 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-09 15:48:04,014 >> tokenizer config file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-400/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-09 15:48:04,014 >> Special tokens file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tmp-checkpoint-400/special_tokens_map.json
02/09/2024 15:48:56 - INFO - llmtuner.extras.callbacks - {'loss': 0.7704, 'learning_rate': 8.8100e-06, 'epoch': 2.59}
{'loss': 0.7704, 'learning_rate': 8.810015400790994e-06, 'epoch': 2.59}
02/09/2024 15:49:48 - INFO - llmtuner.extras.callbacks - {'loss': 0.7807, 'learning_rate': 7.4841e-06, 'epoch': 2.62}
{'loss': 0.7807, 'learning_rate': 7.4841234255076495e-06, 'epoch': 2.62}
02/09/2024 15:50:40 - INFO - llmtuner.extras.callbacks - {'loss': 0.7931, 'learning_rate': 6.2624e-06, 'epoch': 2.66}
{'loss': 0.7931, 'learning_rate': 6.2624448452974885e-06, 'epoch': 2.66}
02/09/2024 15:51:32 - INFO - llmtuner.extras.callbacks - {'loss': 0.8076, 'learning_rate': 5.1464e-06, 'epoch': 2.69}
{'loss': 0.8076, 'learning_rate': 5.146355805285452e-06, 'epoch': 2.69}
02/09/2024 15:52:24 - INFO - llmtuner.extras.callbacks - {'loss': 0.8155, 'learning_rate': 4.1371e-06, 'epoch': 2.72}
{'loss': 0.8155, 'learning_rate': 4.137113510530544e-06, 'epoch': 2.72}
02/09/2024 15:53:16 - INFO - llmtuner.extras.callbacks - {'loss': 0.7585, 'learning_rate': 3.2359e-06, 'epoch': 2.75}
{'loss': 0.7585, 'learning_rate': 3.2358548098621932e-06, 'epoch': 2.75}
02/09/2024 15:54:08 - INFO - llmtuner.extras.callbacks - {'loss': 0.8192, 'learning_rate': 2.4436e-06, 'epoch': 2.78}
{'loss': 0.8192, 'learning_rate': 2.4435949152906145e-06, 'epoch': 2.78}
02/09/2024 15:55:00 - INFO - llmtuner.extras.callbacks - {'loss': 0.8033, 'learning_rate': 1.7612e-06, 'epoch': 2.82}
{'loss': 0.8033, 'learning_rate': 1.7612262584335237e-06, 'epoch': 2.82}
02/09/2024 15:55:52 - INFO - llmtuner.extras.callbacks - {'loss': 0.7904, 'learning_rate': 1.1895e-06, 'epoch': 2.85}
{'loss': 0.7904, 'learning_rate': 1.1895174852472157e-06, 'epoch': 2.85}
02/09/2024 15:56:44 - INFO - llmtuner.extras.callbacks - {'loss': 0.7832, 'learning_rate': 7.2911e-07, 'epoch': 2.88}
{'loss': 0.7832, 'learning_rate': 7.291125901946027e-07, 'epoch': 2.88}
02/09/2024 15:57:36 - INFO - llmtuner.extras.callbacks - {'loss': 0.8096, 'learning_rate': 3.8053e-07, 'epoch': 2.91}
{'loss': 0.8096, 'learning_rate': 3.805301908254455e-07, 'epoch': 2.91}
02/09/2024 15:58:28 - INFO - llmtuner.extras.callbacks - {'loss': 0.7724, 'learning_rate': 1.4416e-07, 'epoch': 2.94}
{'loss': 0.7724, 'learning_rate': 1.4416294358582384e-07, 'epoch': 2.94}
02/09/2024 15:59:21 - INFO - llmtuner.extras.callbacks - {'loss': 0.8264, 'learning_rate': 2.0277e-08, 'epoch': 2.98}
{'loss': 0.8264, 'learning_rate': 2.0277101514987184e-08, 'epoch': 2.98}
[INFO|trainer.py:1962] 2024-02-09 15:59:52,454 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)


02/09/2024 15:59:52 - INFO - llmtuner.extras.callbacks - {'loss': 0.0000, 'learning_rate': 0.0000e+00, 'epoch': 3.00}
{'train_runtime': 4878.9461, 'train_samples_per_second': 6.149, 'train_steps_per_second': 0.096, 'train_loss': 0.8480870586175185, 'epoch': 3.0}
[INFO|trainer.py:2936] 2024-02-09 15:59:52,457 >> Saving model checkpoint to saves/LLaMA-7B/lora/train_2024-02-09-14-34-22
[INFO|configuration_utils.py:729] 2024-02-09 15:59:52,764 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 15:59:52,765 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

[INFO|tokenization_utils_base.py:2433] 2024-02-09 15:59:52,806 >> tokenizer config file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/tokenizer_config.json
[INFO|tokenization_utils_base.py:2442] 2024-02-09 15:59:52,806 >> Special tokens file saved in saves/LLaMA-7B/lora/train_2024-02-09-14-34-22/special_tokens_map.json
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     0.8481
  train_runtime            = 1:21:18.94
  train_samples_per_second =      6.149
  train_steps_per_second   =      0.096
[INFO|modelcard.py:452] 2024-02-09 15:59:52,811 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
[INFO|tokenization_utils_base.py:2027] 2024-02-09 16:02:38,451 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.model
[INFO|tokenization_utils_base.py:2027] 2024-02-09 16:02:38,451 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2027] 2024-02-09 16:02:38,451 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/special_tokens_map.json
[INFO|tokenization_utils_base.py:2027] 2024-02-09 16:02:38,451 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer_config.json
[INFO|tokenization_utils_base.py:2027] 2024-02-09 16:02:38,451 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/tokenizer.json
[INFO|configuration_utils.py:729] 2024-02-09 16:02:38,674 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/config.json
[INFO|configuration_utils.py:792] 2024-02-09 16:02:38,675 >> Model config LlamaConfig {
  "_name_or_path": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.37.2",
  "use_cache": true,
  "vocab_size": 32000
}

02/09/2024 16:02:38 - INFO - llmtuner.model.patcher - Quantizing model to 4 bit.
[INFO|modeling_utils.py:3476] 2024-02-09 16:02:38,679 >> loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/pytorch_model.bin.index.json
[INFO|modeling_utils.py:1426] 2024-02-09 16:02:38,679 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
[INFO|configuration_utils.py:826] 2024-02-09 16:02:38,680 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}

[INFO|modeling_utils.py:3615] 2024-02-09 16:02:38,786 >> Detected 4-bit loading: activating 4-bit loading for this model
Loading checkpoint shards: 100% 2/2 [00:05<00:00,  2.63s/it]
[INFO|modeling_utils.py:4350] 2024-02-09 16:02:44,247 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4358] 2024-02-09 16:02:44,247 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at elyza/ELYZA-japanese-Llama-2-7b-instruct.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:781] 2024-02-09 16:02:44,344 >> loading configuration file generation_config.json from cache at /root/.cache/huggingface/hub/models--elyza--ELYZA-japanese-Llama-2-7b-instruct/snapshots/48fa08b3098a23d3671e09565499a4cfbaff1923/generation_config.json
[INFO|configuration_utils.py:826] 2024-02-09 16:02:44,344 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "use_cache": false
}

02/09/2024 16:02:44 - INFO - llmtuner.model.adapter - Fine-tuning method: LoRA
02/09/2024 16:02:44 - INFO - llmtuner.model.adapter - Loaded adapter(s): saves/LLaMA-7B/lora/train_2024-02-09-14-34-22
02/09/2024 16:02:44 - INFO - llmtuner.model.loader - trainable params: 0 || all params: 6742609920 || trainable%: 0.0000
02/09/2024 16:02:44 - INFO - llmtuner.model.loader - This IS expected that the trainable params is 0 if you are using model for inference only.
Keyboard interruption in main thread... closing server.
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/gradio/blocks.py", line 2361, in block_thread
    time.sleep(0.1)
KeyboardInterrupt

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/content/LLaMA-Factory/src/train_web.py", line 11, in <module>
    main()
  File "/content/LLaMA-Factory/src/train_web.py", line 7, in main
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)
  File "/usr/local/lib/python3.10/dist-packages/gradio/blocks.py", line 2266, in launch
    self.block_thread()
  File "/usr/local/lib/python3.10/dist-packages/gradio/blocks.py", line 2365, in block_thread
    self.server.close()
  File "/usr/local/lib/python3.10/dist-packages/gradio/networking.py", line 70, in close
    def close(self):
KeyboardInterrupt
Killing tunnel 0.0.0.0:7860 <> https://1880c749478f2359be.gradio.live
```

## LORAをモデルに反映？

https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#merge-lora-weights-and-export-model

```bash
python src/export_model.py \
    --model_name_or_path path_to_llama_model \
    --adapter_name_or_path path_to_checkpoint \
    --template default \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```