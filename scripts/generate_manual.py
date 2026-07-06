# -*- coding: utf-8 -*-
from fpdf import FPDF
from pathlib import Path


class PDF(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.add_font('SimHei', '', 'C:/Windows/Fonts/simhei.ttf')
        self.add_font('SimHei', 'B', 'C:/Windows/Fonts/simhei.ttf')
        self.set_auto_page_break(True, 20)
        self.L = self.l_margin
        self.W = self.w - self.l_margin - self.r_margin

    def header(self):
        if self.page_no() > 1:
            self.set_font('SimHei', '', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 6, align='C', text='电信客户流失预测 - v0.2 使用说明')
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('SimHei', '', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, align='C', text=f'第 {self.page_no()} 页')

    def h1(self, t):
        self.ln(6); self.set_x(self.L)
        self.set_font('SimHei', 'B', 18); self.set_text_color(30, 60, 120)
        self.cell(0, 12, align='C', text=t); self.ln(16)

    def h2(self, t):
        self.ln(3); self.set_x(self.L)
        self.set_font('SimHei', 'B', 13); self.set_text_color(30, 60, 120)
        self.cell(0, 10, text=t); self.ln(13)

    def h3(self, t):
        self.set_x(self.L)
        self.set_font('SimHei', 'B', 11); self.set_text_color(50, 80, 140)
        self.cell(0, 8, text=t); self.ln(10)

    def p(self, t):
        self.set_x(self.L); self.set_font('SimHei', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(self.W, 6.5, text=t); self.set_x(self.L)

    def li(self, t):
        self.set_x(self.L + 6); self.set_font('SimHei', '', 10)
        self.set_text_color(40, 40, 40)
        self.cell(5, 6.5, text='-')
        self.multi_cell(self.W - 11, 6.5, text=t); self.set_x(self.L)

    def cmd(self, t):
        self.set_x(self.L); self.set_fill_color(245, 245, 245)
        self.set_text_color(60, 60, 60); self.set_font('SimHei', '', 9)
        for line in t.strip().split('\n'):
            self.set_x(self.L); self.cell(self.W, 6, fill=True, text=f'  {line}'); self.ln()
        self.ln(3); self.set_x(self.L)

    def tip(self, t):
        self.set_x(self.L); self.set_fill_color(230, 240, 255)
        self.set_text_color(30, 60, 120); self.set_font('SimHei', '', 9)
        self.multi_cell(self.W, 6, fill=True, text=f'[提示] {t}')
        self.ln(3); self.set_x(self.L)

    def warn(self, t):
        self.set_x(self.L); self.set_fill_color(255, 243, 205)
        self.set_text_color(100, 80, 20); self.set_font('SimHei', '', 9)
        self.multi_cell(self.W, 6, fill=True, text=f'[注意] {t}')
        self.ln(3); self.set_x(self.L)

    def tbl(self, hdrs, rows, wids=None):
        self.set_x(self.L)
        if wids is None:
            wids = [self.W / len(hdrs)] * len(hdrs)
        self.set_fill_color(30, 60, 120); self.set_text_color(255, 255, 255)
        self.set_font('SimHei', 'B', 9)
        for i, h in enumerate(hdrs):
            self.cell(wids[i], 8, border=1, fill=True, align='C', text=h)
        self.ln()
        for ri, row in enumerate(rows):
            c = 245 if ri % 2 == 0 else 255
            self.set_fill_color(c, c, c)
            self.set_text_color(40, 40, 40); self.set_font('SimHei', '', 9)
            for i, cell in enumerate(row):
                self.cell(wids[i], 7, border=1, fill=True, align='C', text=str(cell))
            self.ln()
        self.ln(4); self.set_x(self.L)


def build():
    pdf = PDF()
    root = Path(__file__).resolve().parent.parent

    # ===== 一 =====
    pdf.add_page()
    pdf.h1('一、项目介绍')
    pdf.h2('1.1 做什么的')
    pdf.p('用电脑分析电信客户数据，找出哪些客户可能会取消服务(流失)。使用逻辑回归、随机森林、XGBoost 三种方法，比较哪种预测最准，最后生成图表。')
    pdf.h2('1.2 为什么做')
    pdf.p('医疗大数据综合应用实验课程期末项目。电信公司如果提前发现高风险客户就可以挽留，降低运营成本。')
    pdf.h2('1.3 当前版本 v0.2')
    pdf.li('数据清洗：7043 条记录，修复缺失值')
    pdf.li('EDA 探索分析：多角度观察数据，发现关键因素')
    pdf.li('特征工程 + SMOTE：解决数据不均衡问题')
    pdf.li('三模型 + GridSearchCV 自动调优')
    pdf.li('消融实验：量化各模块贡献')
    pdf.li('46 个单元测试，70% 代码覆盖')
    pdf.h2('1.4 数据')
    pdf.p('Kaggle 公开数据集 Telco Customer Churn。7043 条，20 个特征：性别、年龄、入网时长、服务订阅、合同类型、支付方式、月费/总费等。不含个人隐私。')

    # ===== 二 =====
    pdf.add_page()
    pdf.h1('二、跑起来')
    pdf.warn('面向零编程基础的同学，按步骤复制粘贴命令即可，不需要任何编程知识。')

    pdf.h2('第一步：安装 Python')
    pdf.p('检查是否已有 Python。按 Win+R，输入 cmd 回车，在黑色窗口输入：')
    pdf.cmd('python --version')
    pdf.p('看到 Python 3.x.x 说明已有，跳到第二步。否则：')
    pdf.li('打开 https://www.python.org/downloads/')
    pdf.li('下载安装包，双击运行')
    pdf.li('安装时务必勾选 Add Python to PATH')
    pdf.warn('Add Python to PATH 这一步最关键，不勾选后续命令无法识别 python。')

    pdf.h2('第二步：进入项目目录')
    pdf.p('方法一(推荐)：Win+R 打开 cmd，输入 cd 加空格，把项目文件夹拖进窗口，回车。')
    pdf.p('方法二：打开项目文件夹，点击地址栏，输入 cmd 回车。')
    pdf.p('确认：输入 dir 回车，应看到 src、data、README.md 等。')

    pdf.h2('第三步：安装依赖')
    pdf.cmd('pip install -r requirements.txt')
    pdf.p('等待 2-5 分钟，看到 Successfully installed 即完成。')
    pdf.tip('速度慢换清华源：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple')
    pdf.tip('项目自带中文字体(assets/fonts/NotoSansSC.ttf)，任何电脑图表中文都能正常显示。')

    pdf.h2('检查数据文件')
    pdf.p('确认 data/raw/ 下有 WA_Fn-UseC_-Telco-Customer-Churn.csv。没有的话从 Kaggle 下载放入。')
    pdf.warn('文件名必须完全一致，不要改名或加后缀。')

    # ===== 三 =====
    pdf.add_page()
    pdf.h1('三、运行')

    pdf.h2('3.1 完整运行(约3-5分钟)')
    pdf.cmd('python src/main.py')
    pdf.p('自动完成：加载数据 -> 清洗 -> 特征工程 -> SMOTE -> 训练 -> GridSearchCV 调优 -> 交叉验证 -> 测试评估 -> 出图 -> 存记录。全程自动。')

    pdf.h2('3.2 EDA 图表')
    pdf.cmd('python scripts/eda_charts.py')
    pdf.p('生成 4 张数据探索图到 results/。')

    pdf.h2('3.3 消融实验')
    pdf.cmd('python scripts/ablation.py')
    pdf.p('对比四组配置，输出各模块贡献表。')

    pdf.h2('3.4 单元测试')
    pdf.cmd('python -m pytest tests/ -v')
    pdf.p('46 个测试，全绿 passed 说明一切正常。')

    # ===== 四 =====
    pdf.add_page()
    pdf.h1('四、看懂结果')

    pdf.h2('4.1 命令行输出')
    pdf.h3('加载数据')
    pdf.p('显示 7043 行数据、缺失值数量、填充方式。Churn 占比约 26.5%，流失:留存约 1:3，数据不均衡。')
    pdf.h3('特征工程')
    pdf.p('自动剔除冗余特征(如 TotalCharges 与 MonthlyCharges 高度相关)。SMOTE 过采样：流失样本从 1308 条扩充到 3621 条。')
    pdf.h3('GridSearchCV')
    pdf.p('每个模型自动尝试几十到上百种参数组合，找最优那组。这步最耗时(3-5分钟)。')
    pdf.h3('最终对比表')
    pdf.tbl(
        ['模型', '准确率', '精确率', '召回率', 'F1', 'AUC'],
        [
            ['逻辑回归', '0.7431', '0.5109', '0.7540', '0.6091', '0.8364'],
            ['随机森林', '0.7693', '0.5671', '0.5535', '0.5602', '0.8219'],
            ['XGBoost', '0.7842', '0.5936', '0.5936', '0.5936', '0.8282'],
        ],
        [30, 24, 24, 24, 24, 24],
    )
    pdf.p('通俗解释：')
    pdf.li('准确率 Accuracy：猜对的比例，越高越好')
    pdf.li('精确率 Precision：预测会流失的人里，真的流失了多少')
    pdf.li('召回率 Recall：真的流失的人里，被模型发现了多少 - 本项目最关注')
    pdf.li('F1：精确和召回的综合平衡分')
    pdf.li('AUC：模型区分能力，0.8 不错，0.9 以上优秀')

    pdf.h2('4.2 结果图表')
    pdf.h3('主流程 4 张图(python src/main.py)')
    pdf.li('churn_distribution.png：流失分布饼图')
    pdf.li('feature_importance.png：特征重要性排序，越长越重要')
    pdf.li('roc_curves.png：ROC曲线，越靠左上角越好')
    pdf.li('confusion_matrices.png：预测正确/错误的数量')

    pdf.h3('EDA 4 张图(python scripts/eda_charts.py)')
    pdf.li('eda_churn_distribution.png：流失分布')
    pdf.li('eda_numerical.png：数值特征分组对比')
    pdf.li('eda_categorical.png：类别特征流失率')
    pdf.li('eda_correlation.png：相关性热力图')

    pdf.h2('4.3 实验记录')
    pdf.li('results/experiments.csv：每次运行自动记录(可用Excel打开)')
    pdf.li('results/ablation.csv：消融实验四组对比')

    # ===== 五 =====
    pdf.add_page()
    pdf.h1('五、常见问题')

    pdf.h3('Q1：python 不是内部或外部命令')
    pdf.p('Python 没装或没勾选 Add to PATH。重装并勾选。关掉命令行重开。')

    pdf.h3('Q2：pip install 慢/报错')
    pdf.p('换清华源：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple。不行换手机热点。')

    pdf.h3('Q3：找不到数据文件')
    pdf.p('检查 data/raw/ 下是否有 WA_Fn-UseC_-Telco-Customer-Churn.csv。文件名必须完全一致。不要改名。')

    pdf.h3('Q4：运行时红色报错')
    pdf.p('截图保存错误信息(从最后一行往上截)，联系技术负责人。常见原因：依赖没装全、数据文件损坏、Python版本太低。')

    pdf.h3('Q5：图表中文是方框乱码')
    pdf.p('项目自带 NotoSansSC.ttf 开源字体，任何系统都应正常。如仍异常，重新 clone 完整项目代码。')

    pdf.h3('Q6：想改配置关闭SMOTE等')
    pdf.p('用记事本打开 configs/config.yaml：')
    pdf.li('imbalance.method 改为 none 关闭 SMOTE')
    pdf.li('features.selection_method 改为 none 关闭特征选择')
    pdf.li('grid_search.enabled 改为 false 跳过 GridSearchCV')
    pdf.p('保存后重新运行。改前备份原文件。')

    pdf.h3('Q7：跑一半卡住了')
    pdf.p('GridSearchCV 需 3-5 分钟，屏幕持续输出，正常。其他阶段卡住按 Ctrl+C 停止，检查数据文件后重试。')

    pdf.h3('Q8：结果和别人不一样')
    pdf.p('项目用固定随机种子(seed=42)，配置相同结果应该一样。不一样说明数据或config.yaml不一致。')

    # ===== 附录 =====
    pdf.add_page()
    pdf.h1('附录：文件结构')

    files = [
        ('README.md', '项目总览和快速开始'),
        ('使用说明.pdf', '本文件'),
        ('requirements.txt', 'Python 依赖列表'),
        ('configs/config.yaml', '配置：seed、SMOTE、GridSearchCV'),
        ('data/raw/', '原始数据存放目录'),
        ('src/main.py', '主入口，一键运行'),
        ('src/data.py', '数据加载与清洗'),
        ('src/features.py', '特征工程 + SMOTE'),
        ('src/models.py', '模型训练 + GridSearchCV'),
        ('src/evaluate.py', '评估指标计算'),
        ('src/visualize.py', '图表生成'),
        ('src/tracking.py', '实验记录 CSV'),
        ('scripts/eda_charts.py', 'EDA 图表生成'),
        ('scripts/ablation.py', '消融实验'),
        ('tests/', '46 个单元测试'),
        ('results/', '输出图表 + CSV'),
        ('assets/fonts/', '开源中文字体'),
    ]
    pdf.set_font('SimHei', '', 9); pdf.set_text_color(40, 40, 40)
    for name, desc in files:
        pdf.set_x(pdf.L)
        pdf.set_font('SimHei', 'B', 9); pdf.cell(55, 6.5, text=name)
        pdf.set_font('SimHei', '', 9); pdf.cell(135, 6.5, text=desc)
        pdf.ln()
    pdf.ln(6)

    pdf.h2('术语速查')
    pdf.tbl(
        ['术语', '通俗解释'],
        [
            ['EDA', '建模前先看看数据什么样'],
            ['SMOTE', '自动生成少数样本解决不均衡'],
            ['GridSearchCV', '自动尝试参数组合找最优'],
            ['消融实验', '逐项关功能看贡献多大'],
            ['Recall', '真正流失的人中被发现的比例'],
            ['AUC', '模型区分能力，越接近1越好'],
        ],
        [45, 145],
    )

    out = root / '使用说明.pdf'
    pdf.output(str(out))
    print(f'[PDF] 已生成: {out}')


if __name__ == '__main__':
    build()
