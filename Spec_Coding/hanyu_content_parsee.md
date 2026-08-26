# 需求
根据汉语组词页面结果解析成多拼音，每个拼音获取词语的列表

# 环境
python3、requests、BeautifulSoup

# 代码实现过程
已知：输入url（例如："https://hanyu.baidu.com/edu-web-go/hanyu/search?query=长组词&flag=1&srcid=51382&subScene=WordZuci&length=all&pinyin=cháng&grade=all"）
1、解析出data.filter.pinyin.options中text的值(value)
2、依次将上面的值value分别填入到url的pinyin=，并请求结果
3、解析出 data.termList中的term的值，
4、将步骤3中的term的结果 放在对应的拼音中呢

# 验证
1、输入 https://hanyu.baidu.com/edu-web-go/hanyu/search?query=长组词&flag=1&srcid=51382&subScene=WordZuci&length=all&pinyin=cháng&grade=all
2、解析拉回的结果，找出这个字data.filter.pinyin.options 三个拼音
3、验证每个拼音对应多个词语

如果验证没有通过，如果校验通过不用自行下面过程
1、校验一下url 是否正确
2、解析的字符的层级是否正确
3、重新优化代码
4、返回到验证阶段，重新验证生成的结果，直到验证通过