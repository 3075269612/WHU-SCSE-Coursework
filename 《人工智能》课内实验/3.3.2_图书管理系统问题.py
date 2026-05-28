# 3.3.2 谓词逻辑案例 - 图书管理系统问题

# 定义事实事实，转换为用谓词逻辑表示
facts = {
    "Authored": [("A", "BookA"), ("B", "BookB")],
    "Reads": [("A", "BookA"), ("B", "BookB"), ("C", "BookA")],
    "Borrowed": [("C", "BookA"), ("B", "BookB")]
}

def all_authors_read_own_books(facts):
    """
    问题1：所有的作者都读过自己写的书？
    谓词逻辑表达式：∀x∀y(Authored(x,y) -> Reads(x,y))
    """
    authored = facts["Authored"]
    reads = facts["Reads"]
    # 检查每个作者是否读过他们自己写的书
    for author, book in authored:
        if (author, book) not in reads:
            return False
    return True

def exists_non_author_reads_book(facts):
    """
    问题2：存在非作者读这本书吗？
    谓词逻辑表达式：∃x∃y(Reads(x,y) ∧ ﹁Authored(x,y))
    """
    authors = {author for author, _ in facts["Authored"]}  # A, B
    reads = facts["Reads"]  # [("A", "BookA"), ("B", "BookB"), ("C", "BookA")]
    
    # 由于所有作者都会读自己写的书，检查是否有读者读了书，但他们不是已知的作者
    for reader, _ in reads:
        if reader not in authors:
            return True
    return False

def borrowed_implies_authored(facts):
    """
    问题3：如果一个人借了一本书，那么这个人就是这本书的作者吗？ 
    谓词逻辑表达式：∀x∀y(Borrowed(x,y) -> Authored(x,y))
    """
    borrowers = facts["Borrowed"]
    authors = facts["Authored"]
    
    for x in borrowers:
        if x not in authors:
            return False
    return True

def main():
    # 运行代码并输出结果
    print("问题1：所有的作者都读过自己写的书？", all_authors_read_own_books(facts))
    print("问题2：存在非作者读这本书吗？", exists_non_author_reads_book(facts))
    print("问题3：如果一个人借阅了一本书，那么这个人就是这本书的作者吗？", borrowed_implies_authored(facts))

if __name__ == '__main__':
    main()