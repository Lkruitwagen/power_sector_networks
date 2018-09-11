import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('country_iso_policy_scrape.csv', encoding='utf-8')
    headers = list(df)

    h_split = []

    policy_cols = [h for h in headers if isnum(h.split('_')[0])]

    for h in headers:

        if isnum(h.split('_')[0]):
            h_split.append(h.split('_'))


    print set([h[1] for h in h_split])

    harmon_dict = {
        'SALES & OTHER TAX CREDITS': 'SALES_TAX_CREDIT', 
        'BIOFUELS MANDATE':'BIOFUEL_TRANSPORT_MANDATE', 
        'RPS':'RPS', 
        'TENDERING':'TENDERING', #== public competitive bidding
        'INVESTMENT & TAX CREDITS':'INVESTMENT_TAX_CREDIT', 
        'TRADEABLE RECS':'TRADEABLE_RECS', 
        'FIT':'FIT', 
        'PUBLIC INVESTMENT/FINANCE/GRANTS':'PUBLIC_INVESTMENT', 
        'ENERGY PRODUCTION PAYMENTS':'ENERGY_PRODUCTION_PAYMENTS', 
        'PUBLIC INVESTMENT/FINANCE/GRANTS/SUBSIDIES':'PUBLIC_INVESTMENT', #2017 only - grants, capital subsidies merged with public investment/finance
        'RENEWABLE ENERGY TARGET':'RENEWABLE_ENERGY_TARGET', 
        'RENEWABLE ENERGY NDC':'RE_IN_NDC',
        'TRANSPORT / BIOFUELS MANDATE':'BIOFUEL_TRANSPORT_MANDATE', 
        'PUBLIC INVESTMENT/FINANCE':'PUBLIC_INVESTMENT', 
        'CAPITAL SUBSIDY':'SUBSIDIES_GRANTS', 
        'NET METERING':'NET_METERING', 
        'GRANTS':'SUBSIDIES_GRANTS', #= capital subsidy
        'HEAT MANDATE':'HEAT_MANDATE', 
        'PUBLIC COMPETITIVE BIDDING':'TENDERING', #public tendering 
        'TRADEABLE REC':'TRADEABLE_RECS'
        }



    correct_dict = {
        '1':1.0,
        '2':0.5,
        '3':1.0,
        '36':0.5,
        '4':1.0,
        '46':0.5,
        '5':0.0,
        '7':1.0,
        '76':0.5,
        '8':1.0,
        'e': ''
    }

    for c in policy_cols:
        print df[c].unique()
        df[c].replace(correct_dict, inplace=True)
        df.loc[lambda df: df[c].isnull(),[c]] = 0.0


    headers_aug = []

    for h in policy_cols:
        headers_aug.append(h.split('_')[0]+'_'+harmon_dict[h.split('_')[1]])

    print headers_aug

    df = df.rename(index=str, columns=dict(zip(policy_cols,headers_aug)))

    print list(df)



    for h in h_split:
        h[1] = harmon_dict[h[1]]

    df.to_csv('policy_df.csv', encoding='utf-8')
        


def isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()