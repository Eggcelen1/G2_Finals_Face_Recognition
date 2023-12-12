<div align = "center">
   
# 𝔾𝕣𝕠𝕦𝕡 𝕀𝕀

</div>
<div align = "center">
   
## 𝔽𝕚𝕟𝕒𝕝𝕤: 𝔽𝕒𝕔𝕖 ℝ𝕖𝕔𝕠𝕘𝕟𝕚𝕥𝕚𝕠𝕟

</div>




<p align="center"> 
   <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/b50c98e6-e585-40a4-9b70-08b5a2faca23">
</p>

<p align="center"> 
   <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/d330952b-f7f5-4939-908f-314df846531e">
</p>

<p align="center"> 
    <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/76321fc4-489e-4f20-afa4-8704875f04b5")
</p>
<div align = "center">

## 𝕀𝕟𝕥𝕣𝕠𝕕𝕦𝕔𝕥𝕚𝕠𝕟: 𝔽𝕒𝕞𝕠𝕦𝕤 ℂ𝕙𝕖𝕗𝕤 𝕚𝕟 𝕥𝕙𝕖 𝕎𝕠𝕣𝕝𝕕

</div>
<div align="justify">
&nbsp;&nbsp;&nbsp;&nbsp;The world of gastronomy boasts an array of culinary maestros whose artistry transcends borders and tantalizes taste buds globally. From pioneers of nouvelle cuisine to modern-day culinary revolutionaries, the pantheon of famous chefs is a testament to innovation, creativity, and a relentless pursuit of culinary perfection. Visionaries like Gordon Ramsay, known for his fiery passion and unparalleled expertise, and the late Anthony Bourdain, whose exploration of global cuisines ignited a cultural culinary dialogue, have left an indelible mark on the industry. Meanwhile, culinary virtuosos such as Massimo Bottura, celebrated for his avant-garde approach to Italian cuisine, and Heston Blumenthal, a pioneer of molecular gastronomy, continually push the boundaries of culinary imagination. These luminaries, among many others, stand as beacons of inspiration, shaping the way we perceive, savor, and celebrate food worldwide.

### 𝕌𝕥𝕚𝕝𝕚𝕫𝕚𝕟𝕘 𝕥𝕙𝕖 𝕗𝕠𝕝𝕝𝕠𝕨𝕚𝕟𝕘 𝕔𝕠𝕕𝕖𝕤 𝕡𝕣𝕠𝕧𝕚𝕕𝕖𝕕 𝕓𝕖𝕝𝕠𝕨:
---
</div>
<div align = "center">
   
##  ℂ𝕠𝕕𝕖𝕤: 𝔾𝕚𝕥𝕙𝕦𝕓 𝕒𝕤 𝕊𝕠𝕦𝕣𝕔𝕖
</div>

    !git clone https://github.com/Eggcelen1/G2_Finals_Face_Recognition.git
    !pip install face_recognition
    %cd G2_Finals_Face_Recognition
---
<div align = "center">
   
##  ℂ𝕠𝕕𝕖𝕤: 𝕂𝕟𝕠𝕨𝕟 𝕀𝕞𝕒𝕘𝕖𝕤
</div>

    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2

    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("ref ramsay.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]

    face_2 = face_recognition.load_image_file("ref-Guy Fieri.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]

    face_3 = face_recognition.load_image_file("ref.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]

    face_4 = face_recognition.load_image_file("ref Bobby Flay.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]

    face_5 = face_recognition.load_image_file("ref- Giada De Laurentiis.jpg")
    face_5_encoding = face_recognition.face_encodings(face_5)[0]

    known_face_encodings = [
                        face_1_encoding,
                        face_2_encoding,
                        face_3_encoding,
                        face_4_encoding,
                        face_5_encoding
    ]

    known_face_names = [
                    "Gordon Ramsay",
                    "Guy Fieri",
                    "Jamie Oliver",
                    "Bobby Flay",
                    "Giada De Laurentiis",
    ]
--- 
<div align = "center">
   
##  ℂ𝕠𝕕𝕖𝕤: 𝔽𝕒𝕔𝕖 ℝ𝕖𝕔𝕠𝕘𝕟𝕚𝕥𝕚𝕠𝕟 𝕠𝕗 𝔽𝕒𝕞𝕠𝕦𝕤 ℂ𝕙𝕖𝕗𝕤 𝕚𝕟 𝕥𝕙𝕖 𝕎𝕠𝕣𝕝𝕕

</div>

    file_name = ""
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)

    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
  
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
    name = known_face_names[best_match_index]
    cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
    cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)

    cv2_imshow(unknown_image_to_draw)
---
# 𝕂𝕟𝕠𝕨𝕟: 𝔽𝕒𝕞𝕠𝕦𝕤 ℂ𝕙𝕖𝕗𝕤 𝕚𝕟 𝕥𝕙𝕖 𝕎𝕠𝕣𝕝𝕕
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/33d5b8d5-bffe-451e-9a53-afdc6c70d005")
</p>

  ## 👨‍🍳𝔾𝕠𝕣𝕕𝕠𝕟 ℝ𝕒𝕞𝕤𝕒𝕪
➣ Gordon James Ramsay OBE is a British celebrity chef, restaurateur, television presenter, and writer. His restaurant group, Gordon Ramsay Restaurants, was founded in 1997 and has been awarded seventeen Michelin stars overall and currently holds seven.

---

<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/9b96d19b-3e97-4db6-96fb-2e8b8ecf1814")
</p>


## 👨‍🍳𝕁𝕒𝕞𝕚𝕖 𝕆𝕝𝕚𝕧𝕖𝕣
➣ Jamie Trevor Oliver MBE OSI is a British celebrity chef, restaurateur and cookbook author. He is known for his casual approach to cuisine, which has led him to front numerous television shows and open many restaurants. Oliver reached the public eye when his series The Naked Chef premiered in 1999. 

---

<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/6bce7c18-ae13-41c0-a39e-b9dc20759748")
</p>


## 👨‍🍳𝔹𝕠𝕓𝕓𝕪 𝔽𝕝𝕒𝕪
➣ Robert William Flay is an American celebrity chef, restaurateur, and reality television personality. Flay is the owner and executive chef of several restaurants and franchises, including Bobby's Burger Palace, Bobby's Burgers, and Amalfi

---

<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/f6e947c4-31e4-43e0-9aca-acf33da04a52")
</p>

  ## 👨‍🍳𝔾𝕦𝕪 𝔽𝕚𝕖𝕣𝕚
➣ Guy Ramsay Fieri is an American restaurateur, author, and an Emmy Award winning television presenter. He co-owned three now defunct restaurants in California, licenses his name to restaurants in cities all over the world, and is known for hosting various television series on the Food Network.  

---

<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/b5d4542b-1027-4be2-9145-1053509e7e38")
</p>


## 👩‍🍳𝔾𝕚𝕒𝕕𝕒 𝔻𝕖 𝕃𝕒𝕦𝕣𝕖𝕟𝕥𝕚𝕚𝕤
➣ Giada Pamela De Laurentiis is an Italian American chef, entrepreneur, writer, and television personality. She was the host of Food Network's program called Giada at Home. She also appears regularly as a contributor and guest co-host on NBC's program entitled Today

---

# 👨‍🍳👩‍🍳𝕌𝕟𝕜𝕟𝕠𝕨𝕟: 𝔽𝕒𝕞𝕠𝕦𝕤 ℂ𝕙𝕖𝕗𝕤
➣ These are the chefs who are unknown but famous

## ℂ𝕙𝕖𝕗 # 𝟙🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/dc0c15d9-c018-4945-98e2-198aff9dd67c")
</p>
  
---

 ## ℂ𝕙𝕖𝕗  # 𝟚🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/2f6258f1-ba83-41b2-a963-a0125b9022da")
</p>
  
---

 ## ℂ𝕙𝕖𝕗 # 𝟛🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/778d0958-f6e0-4cab-9d42-791e6f8b67ed")
</p>
  
---

## ℂ𝕙𝕖𝕗 # 𝟜🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/768346b6-b9e9-419c-b02d-addf35295c19")
</p>
  
---

## ℂ𝕙𝕖𝕗 # 𝟝🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/d5bb61c6-3566-4a3f-850e-ee53850de5f7")
</p>
  
---

## ℂ𝕙𝕖𝕗 # 𝟞🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/819a0e42-9010-4f0b-84a2-cff30c5fbd91")
</p>
  
--- 

## ℂ𝕙𝕖𝕗 # 𝟟🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/fa12486d-7ab9-48af-833a-a0e07e4e233d")
</p>
  
---

 ## ℂ𝕙𝕖𝕗 # 𝟠🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/993eb993-29ce-48b9-b4b7-5f10d5e2aba5")
</p>
  
---

## ℂ𝕙𝕖𝕗 # 𝟡🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/d2d20d8c-acb8-4d31-9da3-b268ea0ec586")
</p>
  
-----

 ## ℂ𝕙𝕖𝕗 # 𝟙𝟘🍳
<p align="center"> 
 <img src="https://github.com/Eggcelen1/G2_Finals_Face_Recognition/assets/144224540/d4eb6e06-1147-45b6-ac98-c9879fd3eac7")
</p>

# 🔗ℝ𝕖𝕗𝕖𝕣𝕖𝕟𝕔𝕖𝕤
- https://www.pinterest.ph/pin/192810427789537923/
- https://www.pinterest.ph/pin/314477986494739998/
- https://www.pinterest.ph/pin/28851253840343842/
- https://www.pinterest.ph/pin/278730664413848368/
- https://www.pinterest.ph/pin/484418503680726961/
- https://www.pinterest.ph/pin/148829962670955330/
- https://www.pinterest.ph/pin/93801604731117264/
- https://www.pinterest.ph/search/pins/?q=chef&rs=typed
- https://en.wikipedia.org/wiki/Gordon_Ramsay
- https://en.wikipedia.org/wiki/Jamie_Oliver
- https://en.wikipedia.org/wiki/Bobby_Flay
- https://en.wikipedia.org/wiki/Guy_Fieri
- https://en.wikipedia.org/wiki/Giada_De_Laurentiis













