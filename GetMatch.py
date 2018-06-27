# -*-coding:utf-8 -*-
import csv
import dota2api
api = dota2api.Initialise("1A5E194B75F7B137B529ED9F31C7C3C4")
out = open('.\match.csv','w',newline='')
csv_write = csv.writer(out,dialect='excel')
m_id = 3876088500
count = 1
summ = 1
while count <= 100:
	try:
		match = api.get_match_details(match_id=m_id);
	except:
		m_id = m_id + 1
		continue
	else:
		if (match.get('duration') > 900) and (match.get('lobby_type') == 7):

			#csv_write.writerow()
			#print(match)
			playerss = match.get('players')
			li = [summ]
			for player in playerss:
				li.append(player.get('hero_id'))
				#print(player.get('hero_id'))
			li.append(match.get('radiant_win'))
			li.append(match.get('lobby_name'))
			li.append(match.get('match_id'))
			csv_write.writerow(li)
			print(li)
			count = count + 1
			summ = summ + 1
	m_id = m_id + 1


