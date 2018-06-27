# -*-coding:utf-8 -*-
import csv
import dota2api
api = dota2api.Initialise("1A5E194B75F7B137B529ED9F31C7C3C4")
#hist = api.get_match_history(account_id=41231571)
#match = api.get_match_details(match_id=1000193456)
#print(match)
#print(hist)
out = open('.\hero.csv','w',newline='')
csv_write = csv.writer(out,dialect='excel')
hero = api.get_heroes().get('heroes');
d = ['id','hero']
for heros in hero:
	d[0] = heros.get('id')
	d[1] = heros.get('localized_name')
	print(d)
	csv_write.writerow(d)
#print(hero)