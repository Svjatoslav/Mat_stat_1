import math
import numpy as np
from scipy.stats import norm, cauchy, uniform, poisson, laplace
import matplotlib.pyplot as plt
from statistics import median, mean
import pandas as pd


def Distribution_Graphics(distribution, loc=0, scale=1, interval=0.005):
    """НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ"""

    name = distribution.name

    #  выбираем 10 нормально распределенных точек
    dist10 = distribution.rvs(size=10, loc=loc, scale=scale)
    dist50 = distribution.rvs(size=50, loc=loc, scale=scale)
    dist1000 = distribution.rvs(size=1000, loc=loc, scale=scale)

    plt.subplot(311)

    plt.tight_layout()
    # plt.tight_layout(h_pad=0.1)

    #  устанавливаем точки по оси икс
    x = np.linspace(distribution.ppf(interval, loc, scale), distribution.ppf(1 - interval, loc, scale), 100)

    #  плотность распределения:
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, distribution.pdf(x, loc, scale), 'g-', lw=2, alpha=0.6, label='{} pdf'.format(name))

    #  строим гистограмму
    ax1.hist(dist10, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax1.legend('{} pdf'.format(name))
    ax1.set_xlabel('n = 10')
    ax1.set_ylabel('density')

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(x, distribution.pdf(x, loc, scale), 'r-', lw=2, alpha=0.6, label='{} pdf'.format(name))
    ax2.hist(dist50, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax2.set_xlabel('n = 50')
    ax2.set_ylabel('density')

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(x, distribution.pdf(x, loc, scale), 'r-', lw=2, alpha=0.6, label='{} pdf'.format(name))
    ax3.hist(dist1000, density=True, bins='auto', edgecolor='black', alpha=0.5)
    ax3.set_xlabel('n = 1000')
    ax3.set_ylabel('density')
    plt.show()


def z_R(array):
    return (min(array) + max(array)) / 2


def z_P(array, p):
    """подается уже отсторированный массив"""
    if (len(array) * p) % 1 == 0:
        ind = len(array) * p
    else:
        ind = math.floor((len(array) * p)) + 1

    return array[int(ind)]


def z_Q(array):
    return (z_P(array, 0.25) + z_P(array, 0.75)) / 2


def z_Tr(array):
    result = 0
    r = math.floor(len(array) / 4)
    for i in range(r, len(array) - r - 1, 1):
        result += array[i]

    return result / (len(array) - 2 * r)


def Characteristics_Evaluate(distribution, loc=0, scale=1):
    super_dict = {'10': {'mean': 0,
                         'median': 0,
                         'z_R': 0,
                         'z_Q': 0,
                         'z_Tr': 0},
                  '100': {'mean': 0,
                          'median': 0,
                          'z_R': 0,
                          'z_Q': 0,
                          'z_Tr': 0},
                  '1000': {'mean': 0,
                           'median': 0,
                           'z_R': 0,
                           'z_Q': 0,
                           'z_Tr': 0}}

    distributions_dict = {'10': [], '100': [], '1000': []}

    super_dict_power_two = {'10': super_dict['10'].copy(),
                            '100': super_dict['100'].copy(),
                            '1000': super_dict['1000'].copy()}

    commands = [mean, median, z_R, z_Q, z_Tr]

    for i in range(1000):
        distributions_dict['10'] = (distribution.rvs(size=10, loc=loc, scale=scale))
        distributions_dict['100'] = distribution.rvs(size=100, loc=loc, scale=scale)
        distributions_dict['1000'] = distribution.rvs(size=1000, loc=loc, scale=scale)

        distributions_dict['10'].sort()
        distributions_dict['100'].sort()
        distributions_dict['1000'].sort()

        if i == 0:
            for cmd in commands:
                for key in super_dict.keys():
                    super_dict[key][cmd.__name__] = cmd(distributions_dict[key])
                    super_dict_power_two[key][cmd.__name__] = cmd(distributions_dict[key]) ** 2

        else:
            for cmd in commands:
                for key in super_dict:
                    super_dict[key][cmd.__name__] = (super_dict[key][cmd.__name__] * i + cmd(distributions_dict[key])) / (i + 1)
                    super_dict_power_two[key][cmd.__name__] = ((super_dict_power_two[key][cmd.__name__] * i +
                                                               cmd(distributions_dict[key])** 2) /
                                                               (i + 1))



    return super_dict, super_dict_power_two









#неправильный вывод (mean при n=1000 слишком далеко от 10)
def poisson_calculate(mu):
    super_dict = {'10': {'mean': 0,
                         'median': 0,
                         'z_R': 0,
                         'z_Q': 0,
                         'z_Tr': 0},
                  '100': {'mean': 0,
                          'median': 0,
                          'z_R': 0,
                          'z_Q': 0,
                          'z_Tr': 0},
                  '1000': {'mean': 0,
                           'median': 0,
                           'z_R': 0,
                           'z_Q': 0,
                           'z_Tr': 0}}
    commands = [mean, median, z_R, z_Q, z_Tr]
    distributions_dict = {'10': [], '100': [], '1000': []}

    super_dict_power_two = {'10': super_dict['10'].copy(),
                            '100': super_dict['100'].copy(),
                            '1000': super_dict['1000'].copy()}

    for i in range(1000):
        distributions_dict['10'] = poisson.rvs(mu, size=10)
        distributions_dict['100'] = poisson.rvs(mu, size=100)
        distributions_dict['1000'] = poisson.rvs(mu, size=1000)

        distributions_dict['10'].sort()
        distributions_dict['100'].sort()
        distributions_dict['1000'].sort()

        if i == 0:
            for cmd in commands:
                for key in super_dict.keys():
                    super_dict[key][cmd.__name__] = cmd(distributions_dict[key])
                    super_dict_power_two[key][cmd.__name__] = cmd(distributions_dict[key]) ** 2

        else:
            for cmd in commands:
                for key in super_dict:
                    super_dict[key][cmd.__name__] = (super_dict[key][cmd.__name__] * i + cmd(distributions_dict[key])) / (
                                i + 1)
                    super_dict_power_two[key][cmd.__name__] = ((super_dict_power_two[key][cmd.__name__] * i +
                                                               cmd(distributions_dict[key])** 2) /
                                                               (i + 1))
    print(super_dict)
    print(super_dict_power_two)



def write_in_file(super_dict, super_dict_power_two):
    df = pd.DataFrame.from_dict(super_dict, 'index').reset_index()
    df1 = pd.DataFrame.from_dict(super_dict_power_two,'index').reset_index()
    return df, df1



def write_concat_df_in_file():
    df_array = []
    frames = []
    frames_power = []
    distr_array = [[norm, 0, 1], [cauchy, 0, 1], [laplace, 0, 1 / math.sqrt(2)],[uniform, -math.sqrt(3),math.sqrt(3)]]
    count = 0
    for sub_arr in distr_array:
        super_dict, super_dict_power_two = Characteristics_Evaluate(sub_arr[0], sub_arr[1], sub_arr[2])
        sup_dict,super_dict_power_two = write_in_file(super_dict, super_dict_power_two)
        # df2 = pd.DataFrame(np.insert(sup_dict.values, count, values=[f'{sub_arr[0].name}', '', '', '', '', ''], axis=0))
        df2 = pd.DataFrame(super_dict)
        df2_power = pd.DataFrame(np.insert(super_dict_power_two.values, count, values=[f'{sub_arr[0].name}', '', '', '', '', ''], axis=0))
        df2.columns = sup_dict.columns
        df2[['mean','median','z_R','z_Q','z_Tr']] = df2[['mean','median','z_R','z_Q','z_Tr']].astype(int)
        df2_power.columns = super_dict_power_two.columns
        frames.append(df2)
        frames_power.append(df2_power)
    result = pd.concat(frames)
    result_power = pd.concat(frames_power)
    result_power = result_power.apply(lambda x: x*x)
    df_array.append(result)
    df_array.append(result_power)
    multiple_dfs(df_array,'sheet1','file.xlsx',3)

def multiple_dfs(df_list, sheets, file_name, spaces):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')
    column = 0
    for dataframe in df_list:
        dataframe.to_excel(writer,sheet_name=sheets,startrow=0 , startcol=column,index=False)
        column = column + 8
    writer.save()

# list of dataframes

# run function


def emperic_functions(distribution, loc, scale):
    dist20 = distribution.rvs(size=20, loc=loc, scale=scale)
    dist60 = distribution.rvs(size=60, loc=loc, scale=scale)
    dist100 = distribution.rvs(size=100, loc=loc, scale=scale)


    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(dist20,alpha=1, histtype='step', cumulative=True, bins=len(dist20),density=True,range=(-4,4))
    x = np.linspace(-4,4,100)
    ax1.plot(x, distribution(loc=loc,scale=scale).cdf(x),
            'g-', lw=2, alpha=0.8, label='laplace pdf')
    # sns.kdeplot(dist20, cumulative=True)
    ax1.set_box_aspect(1)
    ax1.set_xlim(-4, 4)
    # ax1.legend('{} pdf'.format(name))
    ax1.set_xlabel(f'n = 20 - {distribution.name}')
    # ax1.set_ylabel('density')

    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(dist60,alpha=1, histtype='step', cumulative=True, bins=len(dist60), density=True,range=(-4,4))
    ax2.plot(x, distribution(loc=loc,scale=scale).cdf(x),
            'g-', lw=2, alpha=0.8, label='laplace pdf')
    # sns.kdeplot(dist60, cumulative=True)
    ax2.set_box_aspect(1)
    ax2.set_xlim(-4, 4)
    ax2.set_xlabel(f'n = 60 - {distribution.name}')
    # ax2.set_ylabel('density')

    ax3 = plt.subplot(1, 3, 3)

    ax3.hist(dist100, alpha=1, histtype='step', cumulative=True, bins=len(dist100), density=True,range=(-4,4))
    ax3.plot(x, distribution(loc=loc,scale=scale).cdf(x),
            'g-', lw=2, alpha=0.8, label='laplace pdf')
    ax3.set_xlim(-4,4)
    # sns.kdeplot(dist100, cumulative=True)
    ax3.set_box_aspect(1)
    ax3.set_xlabel(f'n = 100 - {distribution.name}')
    # ax3.set_ylabel('density')
    plt.show()


def emperic_functions_poisson():
    dist20 = poisson.rvs(10, size=20)
    dist60 = poisson.rvs(10, size=60)
    dist100 = poisson.rvs(10, size=100)


    ax1 = plt.subplot(1, 3, 1)
    ax1.hist(dist20,alpha=1, histtype='step', cumulative=True, bins=len(dist20),density=True,range=(6,14))
    x = np.linspace(6,14,100)
    ax1.plot(x, poisson(10).cdf(x),
            'g-', lw=2, alpha=0.8, label='laplace pdf')
    # sns.kdeplot(dist20, cumulative=True)
    ax1.set_box_aspect(1)
    ax1.set_xlim(6, 14)
    # ax1.legend('{} pdf'.format(name))
    ax1.set_xlabel(f'n = 20 - {poisson.name}')
    # ax1.set_ylabel('density')

    ax2 = plt.subplot(1, 3, 2)
    ax2.hist(dist60,alpha=1, histtype='step', cumulative=True, bins=len(dist60), density=True, range=(6, 14))
    x = np.linspace(6, 14, 100)
    ax2.plot(x, poisson(10).cdf(x),
             'g-', lw=2, alpha=0.8, label='laplace pdf')
    # sns.kdeplot(dist20, cumulative=True)
    ax2.set_box_aspect(1)
    ax2.set_xlim(6, 14)
    # ax1.legend('{} pdf'.format(name))
    ax2.set_xlabel(f'n = 60 - {poisson.name}')
    # ax1.set_ylabel('density')

    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(dist100,alpha=1, histtype='step', cumulative=True, bins=len(dist100), density=True, range=(6, 14))
    x = np.linspace(6, 14, 100)
    ax3.plot(x, poisson(10).cdf(x),
             'g-', lw=2, alpha=0.8, label='laplace pdf')
    # sns.kdeplot(dist20, cumulative=True)
    ax3.set_box_aspect(1)
    ax3.set_xlim(6, 14)
    # ax1.legend('{} pdf'.format(name))
    ax3.set_xlabel(f'n = 100 - {poisson.name}')
    # ax1.set_ylabel('density')

    plt.show()

    plt.show()



    plt.show()




if __name__ == '__main__':
    Distribution_Graphics(cauchy)
    # Characteristics_Evaluate(norm)
    # poisson_calculate(10)
    # write_concat_df_in_file()
    # emperic_functions(norm,0,1)
    # emperic_functions(cauchy, 0, 1)
    # emperic_functions(laplace, 0, 1 / math.sqrt(2))
    # emperic_functions(uniform, -math.sqrt(3), 2*math.sqrt(3))
    # emperic_functions_poisson()




