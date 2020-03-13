import datetime

def percent():
    ndays = 9
    base = datetime.date.today()
    date_list = [base - datetime.timedelta(days=x) for x in range(ndays)]
    date_list.reverse()
    print(len(date_list))
    print(date_list)
    death=[170.0, 213.0,259.0,304.0,361.0,429.0,493.0,564.0,723.0]
    found=[7821,9800,11880,14401,17238,20471,24441,28605,34598]
    for i in range(len(date_list)):
        percent = death[i]/found[i]*100
        print(str(date_list[i]),":", death[i],"死/确诊{}".format(found[i]),"=","%.2f" % round(percent,2),"%")

if __name__ == '__main__':
    percent()