# 重复商户面板中的降价收益分布资料约定

经营日数不是权重，样本权重不是采购量。分层和簇号描述抽样设计，同一商户在不同月份保持身份。采购表只反映已观测经营月份，缺行须结合经营资格与回收状态解释。参考价的品规与采购品规对应，节省额可以为负。月份采用公历整月，退出与未回收分开保存，不允许默认为相同状态。

## 数据表

### `merchants`

退出月为不再经营的首月，入场月为首个经营月，不完整观察不同于退出。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `group` | `text` |
| `entry_month` | `text` |
| `exit_month` | `text?` |

### `purchases`

采购公斤数与元每公斤价格，支出由数量乘单价形成；未观察月没有虚构零采购行。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `merchant_id` | `text` |
| `month` | `text` |
| `species` | `text` |
| `grade` | `text` |
| `quantity_kg` | `number` |
| `unit_price` | `money` |

### `price_reference`

同品规参考价格，可用于固定篮子比较，不声称是未经价格变化的反事实。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `month` | `text` |
| `species` | `text` |
| `grade` | `text` |
| `unit_price` | `money` |
| `reference_kind` | `text` |

### `panel_design`

月度总体权重、回收标记、实际经营日数及抽样层簇，商户跨月重复观测。

| 字段 | 解析类型 |
| --- | --- |
| `id` | `text` |
| `merchant_id` | `text` |
| `month` | `text` |
| `weight` | `number` |
| `observed` | `bool` |
| `days_open` | `int` |
| `stratum` | `text` |
| `cluster` | `text` |

## 文件和数值语义

空单元格只在带问号的类型中表示缺失；零和空值不同。金额由十进制定点字符串保存，浮点数仅用于探索性统计。所有数据均为本地生成的虚构经营记录，不对应真实公司、客户或住址。数值剖面只反映观察数据，不能代替总体推断。

研究备注的参数为业务自由对象，不代表已执行了任何统计模型。原始表与已有视图可单独查询。
