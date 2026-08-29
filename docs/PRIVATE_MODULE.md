# 私有第一模块与 Git 边界

## 1. 私有目录

以下目录仅保存在本地：

```text
pipeline/modules/module_01_local_synopsis_private/
tests/private/
```

其中包括梗概提示词、解析器、合并器、质量门槛和真实语料审计。旧事实优先实现不属于第一模块。

## 2. 公共仓库保留内容

公共仓库必须保留：

- `pipeline/contracts/synopsis.py`；
- `pipeline/contracts/outline.py`；
- `pipeline/contracts/fact_hydration.py`；
- 调度器的阶段接口；
- `config.example.yaml`；
- 第一模块安装和产物说明；
- 不含真实小说的合成契约测试。

## 3. 动态加载

公共调度器从配置读取：

```yaml
modules:
  local_synopsis:
    entrypoint: pipeline.modules.module_01_local_synopsis_private.api:run
```

模块不存在时必须明确报告“未安装私有梗概提取模块”，不能在导入 `pipeline` 时直接崩溃。

## 4. Git 操作

`.gitignore`忽略私有代码、语料、密钥、运行产物和输出。若目录以前已经被跟踪，需要使用 `git rm --cached`停止跟踪；该操作不得删除本地文件。

公开提交前必须检查：

```text
git status --short
git ls-files pipeline/modules/module_01_local_synopsis_private
git ls-files novels input corpus runs config.yaml
```

上述私有路径不应出现在第二条及第三条命令的结果中。
