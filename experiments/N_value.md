# Check what N-Value for DESM is good

Since there is no alpha Value, first N arguments must be selected by DESM, which further on will be sorted by DPH.
The context space should not be too small, otherwise relevant args could be ignored.
On the other side, if the context space is too big, many arguments that aren't relevant could be chosen.

Experiment: We chose the first 5 topics and for each the top 20 arguments for different N values.
Note: Only the top 1000 Arguments for DPH are chosen, so if the DPH-value for a argument id is not found, it is removed.

Precision: if r > 1


N = 100

### 1 Teachers Get Tenure
---
1. ID: c065954f-2019-04-18T14:32:52Z-00001-000 (PRO) r = 3
2. ID: c065954f-2019-04-18T14:32:52Z-00003-000 (PRO) r = 3
3. ID: b0680508-2019-04-18T13:48:51Z-00002-000 (CON) r = 3
4. ID: eb991d36-2019-04-15T20:24:15Z-00012-000 (CON) r = 1
5. ID: a2ea9045-2019-04-15T20:22:17Z-00015-000 (PRO) r = 0
6. ID: 91988e1b-2019-04-17T11:47:31Z-00047-000 (PRO) r = 2
7. ID: ab3e066e-2019-04-17T11:47:19Z-00040-000 (PRO) r = 0
8. ID: 91988e1b-2019-04-17T11:47:31Z-00049-000 (PRO) r = 2
9. ID: 91988e1b-2019-04-17T11:47:31Z-00032-000 (PRO) r = 3
10. ID: da86b00e-2019-04-15T20:22:17Z-00011-000 (PRO) r = 3
11. ID: 28fce1af-2019-04-18T11:21:06Z-00001-000 (CON) r = 0
12. ID: eef749e-2019-04-18T18:41:19Z-00003-000 (PRO) r = 0
13. ID: 91988e1b-2019-04-17T11:47:31Z-00036-000 (PRO) r = 3
14. ID: 91988e1b-2019-04-17T11:47:31Z-00040-000 (PRO) r = 3
15. ID: eb991d36-2019-04-15T20:24:15Z-00028-000 (CON) r = 1
16. ID: ab3e066e-2019-04-17T11:47:19Z-00032-000 (PRO) r = 0
17. ID: 91988e1b-2019-04-17T11:47:31Z-00044-000 (PRO) r = 3
18. ID: 91988e1b-2019-04-17T11:47:31Z-00048-000 (PRO) r = 2
19. ID: 198e1669-2019-04-18T19:15:21Z-00002-000 (PRO) r = 3
20. ID: 91988e1b-2019-04-17T11:47:31Z-00039-000 (PRO) r = 2

NDCG: [1. 1. 1. 0.85589068 0.74359348 0.70963752 0.64461667 0.62737744 0.65374263 0.70126614 0.68262772 0.66549208 0.70521422 0.75354043 0.75537474 0.75537474 0.80535039 0.82637523 0.87459323 0.89492692]
Precision: 0.65

### 2 Vaping E-Cigarettes Safe
---
1. ID: b83fa829-2019-04-18T16:06:32Z-00003-000 (PRO) r = 3
2. ID: 9fb05020-2019-04-18T17:57:33Z-00007-000 (PRO) r = 0
3. ID: 969c6467-2019-04-18T19:17:45Z-00003-000 (PRO) r = 0
4. ID: a405fed6-2019-04-18T14:24:45Z-00003-000 (PRO) r = 0
5. ID: d291cb2f-2019-04-18T11:24:40Z-00002-000 (CON) r = 0
6. ID: 18710bc8-2019-04-18T16:37:00Z-00000-000 (CON) r = 3
7. ID: 7916b405-2019-04-18T18:52:54Z-00005-000 (CON) r = 0
8. ID: b2b2d4a9-2019-04-18T15:55:06Z-00000-000 (PRO) r = 3
9. ID: 4edcd77f-2019-04-18T16:30:17Z-00005-000 (PRO) r = 0
10. ID: 18710bc8-2019-04-18T16:37:00Z-00001-000 (PRO) r = 2
11. ID: a8e68bc8-2019-04-18T17:52:43Z-00003-000 (PRO) r = 0
12. ID: d291cb2f-2019-04-18T11:24:40Z-00006-000 (CON) r = 0
13. ID: 87b8c230-2019-04-17T11:47:26Z-00139-000 (PRO) r = 0
14. ID: 29e21ae2-2019-04-18T15:05:34Z-00003-000 (PRO) r = 0
15. ID: 71cb6b31-2019-04-18T14:53:18Z-00003-000 (PRO) r = 2
16. ID: de361b6a-2019-04-18T19:51:14Z-00005-000 (PRO) r = 0
17. ID: 6c84f2de-2019-04-18T14:34:39Z-00001-000 (PRO) r = 0
18. ID: e5ccda7-2019-04-17T11:47:44Z-00118-000 (PRO) r = 0
19. ID: 87b8c230-2019-04-17T11:47:26Z-00143-000 (PRO) r = 0
20. ID: 68d8ce13-2019-04-19T12:45:16Z-00014-000 (PRO) r = 0

NDCG: [1. 0.61314719 0.46927873 0.43187115 0.40301463 0.54657134 0.54657134 0.6737083  0.6737083  0.72363574 0.72363574 0.72363574 0.72363574 0.72363574 0.76681588 0.76681588 0.76681588 0.76681588 0.76681588 0.76681588]
Precision: 0.25

-> Problem: Verwechslung mit normalen Rauchen

### 3 Insider Trading Allowed
---
1. ID: 5461331e-2019-04-18T18:07:49Z-00001-000 (PRO) r = 1

NDCG: [1. ]
Precision: 0

### 4 Corporal Punishment Used Schools
---
1. ID: 2fc6200f-2019-04-18T17:01:39Z-00003-000 (CON) r = 3
2. ID: 57c3ac9d-2019-04-18T19:10:29Z-00000-000 (CON) r = 3
3. ID: e7b98175-2019-04-18T14:36:18Z-00004-000 (CON) r = 0
4. ID: cb52628f-2019-04-18T11:53:57Z-00004-000 (PRO) r = 1
5. ID: 91279d46-2019-04-18T17:53:34Z-00001-000 (PRO) r = 2
6. ID: 7da5bbc3-2019-04-18T19:38:37Z-00004-000 (PRO) r = 0
7. ID: 1d10487f-2019-04-17T11:47:31Z-00054-000 (PRO) r = 0
8. ID: 1d10487f-2019-04-17T11:47:31Z-00055-000 (PRO) r = 2
9. ID: 91988e1b-2019-04-17T11:47:31Z-00036-000 (PRO) r = 0
10. ID: 3cb1c9ed-2019-04-18T15:44:50Z-00003-000 (CON) r = 0

NDCG: [1. 1. 0.88386954 0.80349634 0.8851767  0.8851767 0.8851767  0.95178416 0.95178416 0.95178416]
Precision: 0.4

### 5 Social Security Privatized
---
0. ID: 2d6f4e75-2019-04-15T20:22:43Z-00009-000 (PRO) r = 3
1. ID: 2d6f4e75-2019-04-15T20:22:43Z-00007-000 (PRO) r = 3
2. ID: 2d6f4e75-2019-04-15T20:22:43Z-00017-000 (PRO) r = 1
3. ID: cf4c9cbf-2019-04-17T11:47:24Z-00055-000 (PRO) r = 2
4. ID: 2d6f4e75-2019-04-15T20:22:43Z-00011-000 (PRO) r = 1
5. ID: 2d6f4e75-2019-04-15T20:22:43Z-00014-000 (CON) r = 3
6. ID: 2d6f4e75-2019-04-15T20:22:43Z-00012-000 (CON) r = 3
7. ID: 41ee0719-2019-04-18T14:19:05Z-00005-000 (PRO) r = 3
8. ID: cf4c9cbf-2019-04-17T11:47:24Z-00062-000 (PRO) r = 0
9. ID: 36edccb7-2019-04-18T13:24:24Z-00005-000 (CON) r = 3
10. ID: cf4c9cbf-2019-04-17T11:47:24Z-00046-000 (PRO) r = 1
11. ID: cf4c9cbf-2019-04-17T11:47:24Z-00075-000 (PRO) r = 1
12. ID: cf4c9cbf-2019-04-17T11:47:24Z-00042-000 (PRO) r = 1
13. ID: cf4c9cbf-2019-04-17T11:47:24Z-00043-000 (PRO) r = 2
14. ID: cf4c9cbf-2019-04-17T11:47:24Z-00069-000 (PRO) r = 2
15. ID: cf4c9cbf-2019-04-17T11:47:24Z-00044-000 (PRO) r = 1
16. ID: cf4c9cbf-2019-04-17T11:47:24Z-00060-000 (PRO) r = 2
17. ID: cf4c9cbf-2019-04-17T11:47:24Z-00052-000 (PRO) r = 1
18. ID: cf4c9cbf-2019-04-17T11:47:24Z-00078-000 (PRO) r = 1
19. ID: 36edccd6-2019-04-18T12:31:06Z-00001-000 (CON) r = 0

NDCG: [1. 1. 0.79888055 0.73662139 0.6587165  0.69550316 0.76337094 0.82261574 0.79402319 0.84374065 0.84534737 0.84687274 0.84832672 0.86805432 0.88697661 0.88794984 0.90562759 0.90639666 0.91440643 0.91440643]
Precision: 0.5

N = 500

### 1 Teachers Get Tenure
---

1. ID: c065954f-2019-04-18T14:32:52Z-00001-000 (PRO) r = 3
2. ID: c065954f-2019-04-18T14:32:52Z-00003-000 (PRO) r = 3
3. ID: b0680508-2019-04-18T13:48:51Z-00002-000 (CON) r = 3
4. ID: 1a76ed9f-2019-04-18T16:07:27Z-00005-000 (PRO) r = 2
5. ID: 4d8487a-2019-04-18T18:20:20Z-00001-000 (PRO) r = 1
6. ID: ff0947ec-2019-04-18T12:23:12Z-00001-000 (PRO) r = 2
7. ID: 9101cb9d-2019-04-18T16:46:13Z-00000-000 (PRO) r = 0
8. ID: 3f1ddbda-2019-04-18T16:44:26Z-00002-000 (CON) r = 0
9. ID: cafa7193-2019-04-15T20:24:35Z-00011-000 (PRO) r = 0
10. ID: 440fb971-2019-04-18T17:06:22Z-00007-000 (PRO) r = 0
11. ID: 68d82bb6-2019-04-18T19:14:17Z-00003-000 (CON) r = 3
12. ID: 91988e1b-2019-04-17T11:47:31Z-00054-000 (PRO) r = 3
13. ID: 40f19507-2019-04-17T11:47:33Z-00086-000 (PRO) r = 1
14. ID: a2ea9045-2019-04-15T20:22:17Z-00009-000 (PRO) r = 0
15. ID: eb991d36-2019-04-15T20:24:15Z-00012-000 (CON) r = 1
16. ID: a2ea9045-2019-04-15T20:22:17Z-00015-000 (PRO) r = 0
17. ID: 91988e1b-2019-04-17T11:47:31Z-00037-000 (PRO) r = 3
18. ID: cafa7193-2019-04-15T20:24:35Z-00009-000 (PRO) r = 0
19. ID: 1e496b65-2019-04-18T18:17:21Z-00009-000 (PRO) r = 0
20. ID: 5c05ee84-2019-04-18T18:40:28Z-00000-000 (CON) r = 0

NDCG: [1. 1. 1. 0.90392712 0.80407088 0.76359611 0.73195451 0.70433314 0.69597914 0.6881416  0.75599429 0.82889625 0.83901837 0.83901837 0.84865298 0.84865298 0.91334703 0.91334703 0.91334703 0.91334703]
Precision: 0.4

### 2 Vaping E-Cigarettes Safe
---

1. ID: e435a482-2019-04-18T11:12:51Z-00002-000 (CON) r = 3
2. ID: d1d1ca99-2019-04-18T15:06:58Z-00002-000 (CON) r = 0
3. ID: e435a482-2019-04-18T11:12:51Z-00001-000 (PRO) r = 2
4. ID: b83fa829-2019-04-18T16:06:32Z-00000-000 (CON) r = 3
5. ID: b83fa829-2019-04-18T16:06:32Z-00003-000 (PRO) r = 3
6. ID: 8a21ce9-2019-04-18T14:03:04Z-00004-000 (PRO) r = 0
7. ID: 8a21d08-2019-04-18T14:01:36Z-00007-000 (PRO) r = 0
8. ID: 8e48cbbe-2019-04-18T14:00:01Z-00003-000 (PRO) r = 0
9. ID: 8e815158-2019-04-18T20:01:50Z-00001-000 (PRO) r = 0
10. ID: d47da880-2019-04-18T15:10:09Z-00005-000 (PRO) r = 0
11. ID: 9fb05020-2019-04-18T17:57:33Z-00007-000 (PRO) r = 0
12. ID: 4ebdedaf-2019-04-18T11:49:05Z-00001-000 (PRO) r = 0
13. ID: b2b2d4a9-2019-04-18T15:55:06Z-00004-000 (PRO) r = 0
14. ID: e82afa07-2019-04-18T17:50:15Z-00003-000 (PRO) r = 0
15. ID: d4c25be6-2019-04-18T12:50:26Z-00000-000 (PRO) r = 0
16. ID: 4a0dc809-2019-04-18T15:13:53Z-00000-000 (PRO) r = 0
17. ID: 71836ff8-2019-04-18T15:45:50Z-00002-000 (PRO) r = 0
18. ID: a405fed6-2019-04-18T14:24:45Z-00003-000 (PRO) r = 0
19. ID: d291cb2f-2019-04-18T11:24:40Z-00002-000 (CON) r = 0
20. ID: e5ccda7-2019-04-17T11:47:44Z-00118-000 (PRO) r = 0

NDCG: [1. 0.61314719 0.56983845 0.71041176 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232 0.87748232]
Precision: 0.2

### 3 Insider Trading Allowed
---

1. ID: 41e8d87f-2019-04-17T11:47:37Z-00035-000 (PRO) r = 0
2. ID: 5461331e-2019-04-18T18:07:49Z-00001-000 (PRO) r = 1
3. ID: ccc13722-2019-04-15T20:24:25Z-00013-000 (PRO) r = 0

NDCG: [0. 0.63092975 0.63092975]
Precision: 0.0

### 4 Corporal Punishment Used Schools

1. ID: 4712ec0a-2019-04-18T12:53:28Z-00007-000 (CON) r = 3
2. ID: 2fc6200f-2019-04-18T17:01:39Z-00003-000 (CON) r = 3
3. ID: ec0930ea-2019-04-18T18:58:19Z-00003-000 (PRO) r = 2
4. ID: 57c3ac9d-2019-04-18T19:10:29Z-00000-000 (CON) r = 3
5. ID: 8e9af3ad-2019-04-18T17:07:58Z-00002-000 (PRO) r = 1
6. ID: e7b98175-2019-04-18T14:36:18Z-00002-000 (CON) r = 3
7. ID: cb52628f-2019-04-18T11:53:57Z-00002-000 (PRO) r = 1
8. ID: 139da6c8-2019-04-18T18:43:01Z-00005-000 (PRO) r = 2
9. ID: e7b98175-2019-04-18T14:36:18Z-00004-000 (CON) r = 0
10. ID: cb52628f-2019-04-18T11:53:57Z-00004-000 (PRO) r = 1
11. ID: bfc40ccd-2019-04-18T13:22:32Z-00003-000 (PRO) r = 0
12. ID: 91279d46-2019-04-18T17:53:34Z-00001-000 (PRO) r = 2
13. ID: 7da5bbc3-2019-04-18T19:38:37Z-00004-000 (PRO) r = 0
14. ID: 1d10487f-2019-04-17T11:47:31Z-00053-000 (PRO) r = 0
15. ID: 1d10487f-2019-04-17T11:47:31Z-00054-000 (PRO) r = 0
16. ID: 1d10487f-2019-04-17T11:47:31Z-00066-000 (PRO) r = 0
17. ID: fe50a233-2019-04-18T18:12:45Z-00007-000 (PRO) r = 1
18. ID: 1d10487f-2019-04-17T11:47:31Z-00059-000 (PRO) r = 1
19. ID: 66f51d6-2019-04-18T13:50:33Z-00009-000 (PRO) r = 0
20. ID: 71757887-2019-04-18T18:25:21Z-00003-000 (PRO) r = 2

NDCG: [1. 1. 0.86592036 0.88846284 0.85471746 0.93309283 0.90474938 0.90882707 0.89661776 0.89793443 0.88703287 0.91159633 0.90141159 0.90141159 0.90141159 0.90141159 0.91161258 0.92162626 0.92162626 0.95067973]
Precision: 0.4

### 5 Social Security Privatized
---

1. ID: 2d6f4e75-2019-04-15T20:22:43Z-00009-000 (PRO) r = 3
2. ID: 2d6f4e75-2019-04-15T20:22:43Z-00007-000 (PRO) r = 3
3. ID: 2d6f4e75-2019-04-15T20:22:43Z-00013-000 (PRO) r = 2
4. ID: 2d6f4e75-2019-04-15T20:22:43Z-00017-000 (PRO) r = 1
5. ID: 2d6f4e75-2019-04-15T20:22:43Z-00015-000 (PRO) r = 3
6. ID: cf4c9cbf-2019-04-17T11:47:24Z-00055-000 (PRO) r = 2
7. ID: 2d6f4e75-2019-04-15T20:22:43Z-00011-000 (PRO) r = 1
8. ID: 2d6f4e75-2019-04-15T20:22:43Z-00008-000 (CON) r = 3
9. ID: 2d6f4e75-2019-04-15T20:22:43Z-00014-000 (CON) r = 3
10. ID: 2d6f4e75-2019-04-15T20:22:43Z-00010-000 (CON) r = 3
11. ID: 2d6f4e75-2019-04-15T20:22:43Z-00012-000 (CON) r = 3
12. ID: 41ee0719-2019-04-18T14:19:05Z-00005-000 (PRO) r = 3
13. ID: cf4c9cbf-2019-04-17T11:47:24Z-00062-000 (PRO) r = 0
14. ID: 2d6f4e75-2019-04-15T20:22:43Z-00016-000 (CON) r = 2
15. ID: 36edccb7-2019-04-18T13:24:24Z-00005-000 (CON) r = 3
16. ID: cf4c9cbf-2019-04-17T11:47:24Z-00046-000 (PRO) r = 1
17. ID: cf4c9cbf-2019-04-17T11:47:24Z-00075-000 (PRO) r = 1
18. ID: 38b0fe14-2019-04-18T12:54:22Z-00004-000 (CON) r = 2
19. ID: cf4c9cbf-2019-04-17T11:47:24Z-00042-000 (PRO) r = 1
20. ID: cf4c9cbf-2019-04-17T11:47:24Z-00043-000 (PRO) r = 2

NDCG: [1. 1. 0.86592036 0.74435353 0.77789564 0.74024228 0.6855066  0.71060146 0.73107807 0.7764134  0.8177936  0.85583739 0.83545434 0.83918677 0.88435134 0.88517511 0.88597127 0.90026289 0.90092123 0.92040616]
Precision: 0.7

N = 1000

### 1 Teachers get Tenure
---

1. ID: c065954f-2019-04-18T14:32:52Z-00001-000 (PRO) r = 3
2. ID: c065954f-2019-04-18T14:32:52Z-00003-000 (PRO) r = 3
3. ID: b0680508-2019-04-18T13:48:51Z-00002-000 (CON) r = 3
4. ID: ff0947ec-2019-04-18T12:23:12Z-00000-000 (CON) r = 2
5. ID: 24e47090-2019-04-18T19:22:46Z-00004-000 (PRO) r = 3
6. ID: 51530f3f-2019-04-18T18:15:02Z-00004-000 (CON) r = 3
7. ID: e3f07189-2019-04-18T17:54:23Z-00001-000 (CON) r = 1
8. ID: 1a76ed9f-2019-04-18T16:07:27Z-00005-000 (PRO) r = 2
9. ID: e5083ebf-2019-04-18T15:47:48Z-00001-000 (PRO) r = 2
10. ID: 4d8487a-2019-04-18T18:20:20Z-00003-000 (PRO) r = 1
11. ID: 7918bba1-2019-04-18T18:35:21Z-00002-000 (CON) r = 0
12. ID: 4d8487a-2019-04-18T18:20:20Z-00001-000 (PRO) r = 1
13. ID: ff0947ec-2019-04-18T12:23:12Z-00001-000 (PRO) r = 2
14. ID: e43a535a-2019-04-18T11:37:17Z-00001-000 (PRO) r = 1
15. ID: eef749e-2019-04-18T18:41:19Z-00002-000 (CON) r = 0
16. ID: a045c4ae-2019-04-18T16:03:27Z-00003-000 (PRO) r = 0
17. ID: d5189153-2019-04-18T16:33:15Z-00003-000 (PRO) r = 1
18. ID: 1e117ec9-2019-04-18T11:57:50Z-00002-000 (CON) r = 0
19. ID: e3f07189-2019-04-18T17:54:23Z-00002-000 (PRO) r = 1
20. ID: eef749e-2019-04-18T18:41:19Z-00004-000 (CON) r = 2

NDCG: [1. 1. 1. 0.90392712 0.91653237 0.98627788 0.95752373 0.95922319 0.96072274 0.93932337 0.92912951 0.92986686 0.95059077 0.95106814 0.94217709 0.93363578 0.94252201 0.94252201 0.95109572 0.97640464]
Precision: 0.5

### 2 vaping e-cigarettes E-Cigarettes safe
---

1. ID: e435a482-2019-04-18T11:12:51Z-00002-000 (CON) r = 3
2. ID: 65de0e0f-2019-04-18T14:18:27Z-00004-000 (CON) r = 2
3. ID: b62db6e9-2019-04-18T15:15:06Z-00006-000 (CON) r = 0
4. ID: d1d1ca99-2019-04-18T15:06:58Z-00002-000 (CON) r = 0
5. ID: e435a482-2019-04-18T11:12:51Z-00001-000 (PRO) r = 2
6. ID: 6aa773f4-2019-04-18T14:45:16Z-00003-000 (CON) r = 3
7. ID: c5c7b5c5-2019-04-18T13:27:00Z-00003-000 (CON) r = 0
8. ID: b83fa829-2019-04-18T16:06:32Z-00000-000 (CON) r = 3
9. ID: b83fa829-2019-04-18T16:06:32Z-00003-000 (PRO) r = 3
10. ID: 8a21cca-2019-04-18T14:06:51Z-00002-000 (PRO) r = 0
11. ID: 946ce747-2019-04-18T16:43:45Z-00003-000 (PRO) r = 0
12. ID: c81986ed-2019-04-18T15:29:36Z-00003-000 (PRO) r = 0
13. ID: 497a4c74-2019-04-18T16:49:19Z-00004-000 (PRO) r = 3
14. ID: d1d1ca99-2019-04-18T15:06:58Z-00001-000 (PRO) r = 0
15. ID: 8a21ce9-2019-04-18T14:03:04Z-00004-000 (PRO) r = 0
16. ID: 8a21d08-2019-04-18T14:01:36Z-00007-000 (PRO) r = 0
17. ID: 8e48cbbe-2019-04-18T14:00:01Z-00003-000 (PRO) r = 0
18. ID: 29cc3447-2019-04-18T16:26:48Z-00002-000 (CON) r = 0
19. ID: 116a2781-2019-04-18T18:24:09Z-00003-000 (CON) r = 0
20. ID: d291cb2f-2019-04-18T11:24:40Z-00002-000 (CON) r = 0

NDCG: [1. 0.77894125 0.59617097 0.49593822 0.48709935 0.57798476 0.55253166 0.64977801 0.74257461 0.74257461 0.74257461 0.74257461 0.82353992 0.82353992 0.82353992 0.82353992 0.82353992 0.82353992 0.82353992 0.82353992]
Precision: 0.35

### 3 Insider Trading allowed
---

1. ID: 9f34b976-2019-04-18T16:52:39Z-00005-000 (PRO) r = 0
2. ID: 9d3685a4-2019-04-18T19:13:30Z-00002-000 (PRO) r = 0
3. ID: 7cf7eb3b-2019-04-15T20:24:49Z-00016-000 (CON) r = 0
4. ID: 82c81407-2019-04-15T20:22:50Z-00015-000 (PRO) r = 0
5. ID: 41e8d87f-2019-04-17T11:47:37Z-00035-000 (PRO) r = 0
6. ID: 5461331e-2019-04-18T18:07:49Z-00001-000 (PRO) r = 1
7. ID: e7f4127b-2019-04-15T20:22:50Z-00018-000 (PRO) r = 0
8. ID: 40f44a1b-2019-04-17T11:47:22Z-00069-000 (PRO) r = 0
9. ID: ccc13722-2019-04-15T20:24:25Z-00013-000 (PRO) r = 0
10. ID: 27800aae-2019-04-18T16:26:56Z-00002-000 (CON) r = 0
11. ID: 2ffad480-2019-04-18T14:47:13Z-00002-000 (PRO) r = 0
12. ID: 5ace1fff-2019-04-18T15:17:08Z-00003-000 (PRO) r = 0
13. ID: 82c81407-2019-04-15T20:22:50Z-00023-000 (PRO) r = 0
14. ID: 33ab9b2d-2019-04-15T20:24:49Z-00013-000 (PRO) r = 0
15. ID: b58ff37e-2019-04-17T11:47:36Z-00092-000 (PRO) r = 0
16. ID: 846238d1-2019-04-15T20:22:22Z-00013-000 (PRO) r = 0
17. ID: 79953157-2019-04-18T13:02:21Z-00000-000 (PRO) r = 0
18. ID: fae0ed1b-2019-04-15T20:24:11Z-00020-000 (PRO) r = 0
19. ID: 87b8c230-2019-04-17T11:47:26Z-00092-000 (PRO) r = 0

NDCG: [0. 0. 0. 0. 0. 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719 0.35620719]
Precision: 0

### 4 corporal Punishment used Schools
---

1. ID: 4712ec0a-2019-04-18T12:53:28Z-00007-000 (CON) r = 3
2. ID: 2fc6200f-2019-04-18T17:01:39Z-00003-000 (CON) r = 3
3. ID: c6b2791c-2019-04-18T14:59:08Z-00002-000 (CON) r = 3
4. ID: c6b2791c-2019-04-18T14:59:08Z-00004-000 (CON) r = 3
5. ID: c6b278de-2019-04-18T15:01:18Z-00003-000 (CON) r = 3
6. ID: 2fc6200f-2019-04-18T17:01:39Z-00001-000 (CON) r = 2
7. ID: ec0930ea-2019-04-18T18:58:19Z-00003-000 (PRO) r = 2
8. ID: 29b5e1ff-2019-04-18T17:57:40Z-00005-000 (CON) r = 3
9. ID: 29b5e1ff-2019-04-18T17:57:40Z-00001-000 (CON) r = 2
10. ID: cb52628f-2019-04-18T11:53:57Z-00001-000 (CON) r = 3
11. ID: 57c3ac9d-2019-04-18T19:10:29Z-00000-000 (CON) r = 3
12. ID: 8e9af3ad-2019-04-18T17:07:58Z-00002-000 (PRO) r = 1
13. ID: 29b5e1ff-2019-04-18T17:57:40Z-00006-000 (PRO) r = 0
14. ID: e7b98175-2019-04-18T14:36:18Z-00002-000 (CON) r = 3
15. ID: 2fc6200f-2019-04-18T17:01:39Z-00004-000 (PRO) r = 2
16. ID: dab6c791-2019-04-18T12:50:18Z-00001-000 (PRO) r = 2
17. ID: c93845a0-2019-04-18T15:10:52Z-00001-000 (PRO) r = 3
18. ID: 6110d4e2-2019-04-18T18:37:47Z-00001-000 (PRO) r = 3
19. ID: 6e08c139-2019-04-18T17:29:42Z-00000-000 (PRO) r = 2
20. ID: f788467e-2019-04-18T15:05:59Z-00004-000 (PRO) r = 2

NDCG: [1. 1. 1. 1. 1. 0.9384062 0.89169236 0.90033472 0.86695475 0.87541919 0.88262519 0.86974287 0.85035976 0.88188272 0.88428517 0.88654344 0.91370087 0.93938593 0.95152857 0.96892345]
Precision: 0.9

### 5 social Security privatized
---

# Result

The best context size is ... This way most of the DPH top 1000 results can be found 