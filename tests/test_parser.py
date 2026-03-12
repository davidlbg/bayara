from bayara.parser import parse_source


def test_parse_dataset_and_split():
    src = '''
    dataset churn from "data/churn.csv"
    target churn -> exited
    features churn -> age, balance, salary
    split churn test 0.2
    '''
    program = parse_source(src)
    assert len(program.statements) == 4
