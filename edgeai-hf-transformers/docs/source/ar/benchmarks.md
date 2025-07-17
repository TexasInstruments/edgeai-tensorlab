# معايير الأداء
<Tip warning={true}>

أدوات قياس الأداء من Hugging Face أصبحت قديمة،ويُنصح باستخدام مكتبات خارجية لقياس سرعة وتعقيد الذاكرة لنماذج Transformer.

</Tip>

[[open-in-colab]]

لنلق نظرة على كيفية تقييم أداء نماذج 🤗 Transformers، وأفضل الممارسات، ومعايير الأداء المتاحة بالفعل.

يُمكن العثور على دفتر ملاحظات يشرح بالتفصيل كيفية قياس أداء نماذج 🤗 Transformers [هنا](https://github.com/huggingface/notebooks/tree/main/examples/benchmark.ipynb).

## كيفية قياس أداء نماذج 🤗 Transformers

تسمح الفئتان [`PyTorchBenchmark`] و [`TensorFlowBenchmark`] بتقييم أداء نماذج 🤗 Transformers بمرونة. تتيح لنا فئات التقييم قياس الأداء قياس _الاستخدام الأقصى للذاكرة_ و _الوقت اللازم_ لكل من _الاستدلال_ و _التدريب_.

<Tip>

هنا، ييُعرَّف _الاستدلال_ بأنه تمريرة أمامية واحدة، ويتم تعريف _التدريب_ بأنه تمريرة أمامية واحدة وتمريرة خلفية واحدة.

</Tip>

تتوقع فئات تقييم الأداء [`PyTorchBenchmark`] و [`TensorFlowBenchmark`] كائنًا من النوع [`PyTorchBenchmarkArguments`] و [`TensorFlowBenchmarkArguments`]، على التوالي، للتنفيذ. [`PyTorchBenchmarkArguments`] و [`TensorFlowBenchmarkArguments`] هي فئات بيانات وتحتوي على جميع التكوينات ذات الصلة لفئة تقييم الأداء المقابلة. في المثال التالي، يتم توضيح كيفية تقييم أداء نموذج BERT من النوع _bert-base-cased_.

<frameworkcontent>
<pt>
  
```py
>>> from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments

>>> args = PyTorchBenchmarkArguments(models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512])
>>> benchmark = PyTorchBenchmark(args)
```
</pt>
<tf>
  
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments

>>> args = TensorFlowBenchmarkArguments(
...     models=["google-bert/bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> benchmark = TensorFlowBenchmark(args)
```
</tf>
</frameworkcontent>

هنا، يتم تمرير ثلاثة معامﻻت إلى فئات بيانات حجة قياس الأداء، وهي `models` و `batch_sizes` و `sequence_lengths`. المعامل `models` مطلوبة وتتوقع `قائمة` من بمعرّفات النموذج من [مركز النماذج](https://huggingface.co/models) تحدد معامﻻت القائمة `batch_sizes` و `sequence_lengths` حجم `input_ids` الذي يتم قياس أداء النموذج عليه. هناك العديد من المعلمات الأخرى التي يمكن تكوينها عبر فئات بيانات معال قياس الأداء. لمزيد من التفاصيل حول هذه المعلمات، يمكنك إما الرجوع مباشرة إلى الملفات `src/transformers/benchmark/benchmark_args_utils.py`، `src/transformers/benchmark/benchmark_args.py` (لـ PyTorch) و `src/transformers/benchmark/benchmark_args_tf.py` (لـ Tensorflow). أو، بدلاً من ذلك، قم بتشغيل أوامر shell التالية من المجلد الرئيسي لطباعة قائمة وصفية بجميع المعلمات القابلة للتكوين لـ PyTorch و Tensorflow على التوالي.

<frameworkcontent>
<pt>
  
```bash
python examples/pytorch/benchmarking/run_benchmark.py --help
```

يُمكن ببساطة تشغيل كائن التقييم الذي تم تهيئته عن طريق استدعاء `benchmark.run()`.

```py
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.006     
google-bert/bert-base-uncased          8               32            0.006     
google-bert/bert-base-uncased          8              128            0.018     
google-bert/bert-base-uncased          8              512            0.088     
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1227
google-bert/bert-base-uncased          8               32            1281
google-bert/bert-base-uncased          8              128            1307
google-bert/bert-base-uncased          8              512            1539
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 08:58:43.371351
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
  
```bash
python examples/tensorflow/benchmarking/run_benchmark_tf.py --help
```

يُمكن بعد ذلك تشغيل كائن قياس الأداء الذي تم تهيئته عن طريق استدعاء `benchmark.run()`.

```py
>>> results = benchmark.run()
>>> print(results)
>>> results = benchmark.run()
>>> print(results)
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length     Time in s                  
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             0.005
google-bert/bert-base-uncased          8               32            0.008
google-bert/bert-base-uncased          8              128            0.022
google-bert/bert-base-uncased          8              512            0.105
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length    Memory in MB 
--------------------------------------------------------------------------------
google-bert/bert-base-uncased          8               8             1330
google-bert/bert-base-uncased          8               32            1330
google-bert/bert-base-uncased          8              128            1330
google-bert/bert-base-uncased          8              512            1770
--------------------------------------------------------------------------------

====================        ENVIRONMENT INFORMATION         ====================

- transformers_version: 202.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:26:35.617317
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

بشكل افتراضي، يتم تقييم _الوقت_ و _الذاكرة المطلوبة_ لـ _الاستدلال_. في مثال المخرجات أعلاه، يُظهر القسمان الأولان النتيجة المقابلة لـ _وقت الاستدلال_ و _ذاكرة الاستدلال_. بالإضافة إلى ذلك، يتم طباعة جميع المعلومات ذات الصلة حول بيئة الحوسبة، على سبيل المثال نوع وحدة معالجة الرسومات (GPU)، والنظام، وإصدارات المكتبة، وما إلى ذلك، في القسم الثالث تحت _معلومات البيئة_. يمكن حفظ هذه المعلومات بشكل اختياري في ملف _.csv_ عند إضافة المعامل `save_to_csv=True` إلى [`PyTorchBenchmarkArguments`] و [`TensorFlowBenchmarkArguments`] على التوالي. في هذه الحالة، يتم حفظ كل قسم في ملف _.csv_ منفصل. يمكن اختيارًا تحديد مسار كل ملف _.csv_ عبر فئات بيانات معامل قياس الأداء.

بدلاً من تقييم النماذج المدربة مسبقًا عبر معرّف النموذج، على سبيل المثال `google-bert/bert-base-uncased`، يُمكن للمستخدم بدلاً من ذلك قياس أداء تكوين عشوائي لأي فئة نموذج متاحة. في هذه الحالة، يجب إدراج "قائمة" من التكوينات مع معامل قياس الأداء كما هو موضح أدناه.

<frameworkcontent>
<pt>
  
```py
>>> from transformers import PyTorchBenchmark، PyTorchBenchmarkArguments، BertConfig

>>> args = PyTorchBenchmarkArguments(
...     models=["bert-base"، "bert-384-hid"، "bert-6-lay"]، batch_sizes=[8]، sequence_lengths=[8، 32، 128، 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = PyTorchBenchmark(args، configs=[config_base، config_384_hid، config_6_lay])
>>> benchmark.run()
====================       INFERENCE - SPEED - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length       Time in s                  
--------------------------------------------------------------------------------
bert-base                  8              128            0.006
bert-base                  8              512            0.006
bert-base                  8              128            0.018     
bert-base                  8              512            0.088     
bert-384-hid              8               8             0.006     
bert-384-hid              8               32            0.006     
bert-384-hid              8              128            0.011     
bert-384-hid              8              512            0.054     
bert-6-lay                 8               8             0.003     
bert-6-lay                 8               32            0.004     
bert-6-lay                 8              128            0.009     
bert-6-lay                 8              512            0.044
--------------------------------------------------------------------------------

====================      INFERENCE - MEMORY - RESULT       ====================
--------------------------------------------------------------------------------
Model Name             Batch Size     Seq Length      Memory in MB
## نتائج اختبار الأداء

في هذا القسم، يتم قياس _وقت الاستدلال_ و _الذاكرة المطلوبة_ للاستدلال، لمختلف تكوينات `BertModel`. يتم عرض النتائج في جدول، مع تنسيق مختلف قليلاً لكل من PyTorch و TensorFlow.

--------------------------------------------------------------------------------
| اسم النموذج | حجم الدفعة | طول التسلسل | الذاكرة بالميغابايت |
--------------------------------------------------------------------------------
| bert-base | 8 | 8 | 1277 |
| bert-base | 8 | 32 | 1281 |
| bert-base | 8 | 128 | 1307 |
| bert-base | 8 | 512 | 1539 |
| bert-384-hid | 8 | 8 | 1005 |
| bert-384-hid | 8 | 32 | 1027 |
| bert-384-hid | 8 | 128 | 1035 |
| bert-384-hid | 8 | 512 | 1255 |
| bert-6-lay | 8 | 8 | 1097 |
| bert-6-lay | 8 | 32 | 1101 |
| bert-6-lay | 8 | 128 | 1127 |
| bert-6-lay | 8 | 512 | 1359 |
--------------------------------------------------------------------------------

==================== معلومات البيئة ====================

- transformers_version: 2.11.0
- framework: PyTorch
- use_torchscript: False
- framework_version: 1.4.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:35:25.143267
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</pt>
<tf>
  
```py
>>> from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

>>> args = TensorFlowBenchmarkArguments(
...     models=["bert-base", "bert-384-hid", "bert-6-lay"], batch_sizes=[8], sequence_lengths=[8, 32, 128, 512]
... )
>>> config_base = BertConfig()
>>> config_384_hid = BertConfig(hidden_size=384)
>>> config_6_lay = BertConfig(num_hidden_layers=6)

>>> benchmark = TensorFlowBenchmark(args, configs=[config_base, config_384_hid, config_6_lay])
>>> benchmark.run()
==================== نتائج السرعة في الاستدلال ====================
--------------------------------------------------------------------------------
| اسم النموذج | حجم الدفعة | طول التسلسل | الوقت بالثانية |
--------------------------------------------------------------------------------
| bert-base | 8 | 8 | 0.005 |
| bert-base | 8 | 32 | 0.008 |
| bert-base | 8 | 128 | 0.022 |
| bert-base | 8 | 512 | 0.106 |
| bert-384-hid | 8 | 8 | 0.005 |
| bert-384-hid | 8 | 32 | 0.007 |
| bert-384-hid | 8 | 128 | 0.018 |
| bert-384-hid | 8 | 512 | 0.064 |
| bert-6-lay | 8 | 8 | 0.002 |
| bert-6-lay | 8 | 32 | 0.003 |
| bert-6-lay | 8 | 128 | 0.0011 |
| bert-6-lay | 8 | 512 | 0.074 |
--------------------------------------------------------------------------------

==================== نتائج الذاكرة في الاستدلال ====================
--------------------------------------------------------------------------------
| اسم النموذج | حجم الدفعة | طول التسلسل | الذاكرة بالميغابايت |
--------------------------------------------------------------------------------
| اسم النموذج | حجم الدفعة | طول التسلسل | الذاكرة بالميغابايت |
--------------------------------------------------------------------------------
| bert-base | 8 | 8 | 1330 |
| bert-base | 8 | 32 | 1330 |
| bert-base | 8 | 128 | 1330 |
| bert-base | 8 | 512 | 1770 |
| bert-384-hid | 8 | 8 | 1330 |
| bert-384-hid | 8 | 32 | 1330 |
| bert-384-hid | 8 | 128 | 1330 |
| bert-384-hid | 8 | 512 | 1540 |
| bert-6-lay | 8 | 8 | 1330 |
| bert-6-lay | 8 | 32 | 1330 |
| bert-6-lay | 8 | 128 | 1330 |
| bert-6-lay | 8 | 512 | 1540 |
--------------------------------------------------------------------------------

==================== معلومات البيئة ====================

- transformers_version: 2.11.0
- framework: Tensorflow
- use_xla: False
- framework_version: 2.2.0
- python_version: 3.6.10
- system: Linux
- cpu: x86_64
- architecture: 64bit
- date: 2020-06-29
- time: 09:38:15.487125
- fp16: False
- use_multiprocessing: True
- only_pretrain_model: False
- cpu_ram_mb: 32088
- use_gpu: True
- num_gpus: 1
- gpu: TITAN RTX
- gpu_ram_mb: 24217
- gpu_power_watts: 280.0
- gpu_performance_state: 2
- use_tpu: False
```
</tf>
</frameworkcontent>

مرة أخرى، يتم قياس _وقت الاستدلال_ و _الذاكرة المطلوبة_ للاستدلال، ولكن هذه المرة لتكوينات مخصصة لـ `BertModel`. يمكن أن تكون هذه الميزة مفيدة بشكل خاص عند اتخاذ قرار بشأن التكوين الذي يجب تدريب النموذج عليه.

## أفضل الممارسات في اختبار الأداء

يسرد هذا القسم بعض أفضل الممارسات التي يجب مراعاتها عند إجراء اختبار الأداء لنموذج ما.

- حالياً، يتم دعم اختبار الأداء على جهاز واحد فقط. عند إجراء الاختبار على وحدة معالجة الرسوميات (GPU)، يوصى بأن يقوم المستخدم بتحديد الجهاز الذي يجب تشغيل التعليمات البرمجية عليه من خلال تعيين متغير البيئة `CUDA_VISIBLE_DEVICES` في الشل، على سبيل المثال `export CUDA_VISIBLE_DEVICES=0` قبل تشغيل التعليمات البرمجية.
- يجب تعيين الخيار `no_multi_processing` إلى `True` فقط لأغراض الاختبار والتصحيح. ولضمان قياس الذاكرة بدقة، يوصى بتشغيل كل اختبار ذاكرة في عملية منفصلة والتأكد من تعيين `no_multi_processing` إلى `True`.
- يجب دائمًا ذكر معلومات البيئة عند مشاركة نتائج تقييم النموذج. يُمكن أن تختلف النتائج اختلافًا كبيرًا بين أجهزة GPU المختلفة وإصدارات المكتبات، وما إلى ذلك، لذلك فإن نتائج الاختبار بمفردها ليست مفيدة جدًا للمجتمع.

## مشاركة نتائج اختبار الأداء الخاص بك

في السابق، تم إجراء اختبار الأداء لجميع النماذج الأساسية المتاحة (10 في ذلك الوقت) لقياس _وقت الاستدلال_، عبر العديد من الإعدادات المختلفة: باستخدام PyTorch، مع TorchScript وبدونها، باستخدام TensorFlow، مع XLA وبدونه. تم إجراء جميع هذه الاختبارات على وحدات المعالجة المركزية (CPU) (باستثناء XLA TensorFlow) ووحدات معالجة الرسوميات (GPU).

يتم شرح هذا النهج بالتفصيل في [منشور المدونة هذا](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2) وتتوفر النتائج [هنا](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit?usp=sharing).

مع أدوات اختبار الأداء الجديدة، أصبح من الأسهل من أي وقت مضى مشاركة نتائج اختبار الأداء الخاص بك مع المجتمع:

- [نتائج اختبار الأداء في PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/benchmarking/README.md).
- [نتائج اختبار الأداء في TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/benchmarking/README.md).
