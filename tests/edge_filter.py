"""

"""

from icecream import ic

from spider_bot.environments import SpiderBotSimulator

def main():
    zero_case_single()
    zero_case_multiple()
    one_case_single()
    one_case_multiple()
    remove_once()
    remove_at_edge()
    dont_remove_at_edge
    
def zero_case_single():
    data = [
        [0],
        [0],
        [0],
        [0]
    ]
    
    expected = data
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)
    assert expected == result

def zero_case_multiple():
    data = [
        [0 for _ in range(10)],
        [0 for _ in range(10)],
        [0 for _ in range(10)],
        [0 for _ in range(10)]
    ]
    
    expected = data
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)
    assert expected == result
    
def one_case_single():
    data = [
        [1],
        [1],
        [1],
        [1]
    ]
    
    expected = data
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)
    assert expected == result

def one_case_multiple():
    data = [
        [1 for _ in range(10)],
        [1 for _ in range(10)],
        [1 for _ in range(10)],
        [1 for _ in range(10)]
    ]
    
    expected = [
        [0 for _ in range(9)] + [1],
        [0 for _ in range(9)] + [1],
        [0 for _ in range(9)] + [1],
        [0 for _ in range(9)] + [1]
    ]
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)
    assert expected == result
    
def remove_once():
    data = [
        [1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 0, 1]
    ]
    
    expected = [
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1]
    ]
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)

    assert expected == result
    
def remove_at_edge():
    data = [
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1]
    ]
    
    expected = [
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1]
    ]
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)

    assert expected == result
    
def dont_remove_at_edge():
    data = [
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0]
    ]
    
    expected = data
    
    result = SpiderBotSimulator.low_pass_filter(data, 3)

    assert expected == result

if __name__ == '__main__':
    main()