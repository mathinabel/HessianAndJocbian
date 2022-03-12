from PIL import Image

filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\aae with p 30\images\46000.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\aae_without p\images\46800.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\acgan with p 3\images\46800.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\acgan without p\images\44000.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\began without p\images\17200.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\bgan without p\images\18400.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\cgan without p\images\17600.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\ebgan with p 15\images\46800.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\ebgan without p\images\46800.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\info without p\images\varying_c2\46800.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\lsgan with p 15\images\46000.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\sgan without p\images\46400.png'
#filename = r'F:\2021spring\1erciyuangan\gan文章\paperdata\wgan without p\images\28000.png'

img = Image.open(filename)
size = img.size
#print(size)

# 准备将图片切割成100张小图片
weight = int(size[0] // 10)
height = int(size[1] // 10)
# 切割后的小图的宽度和高度
#print(weight, height)

for j in range(10):
    for i in range(10):
        box = (weight * i, height * j, weight * (i + 1), height * (j + 1))
        region = img.crop(box)
        region.save('F:/2021spring/1erciyuangan/gan文章/paperdata/aae with p 30/11/pic2_{}{}.png'.format(j, i))