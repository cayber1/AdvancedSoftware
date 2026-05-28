"""
ContextGuard — Dataset Loader
Proposal §5: "RAG benchmarks (Natural Questions, HotpotQA)"

3-layer strategy (priority order):
  1. data/ folder JSON (downloaded by download_data.py — real dataset)
  2. Hugging Face datasets library (live internet connection)
  3. Built-in fallback samples (offline, 20 HotpotQA + 20 NQ)

Usage:
  from contextguard.data_loader import load_dataset
  items = load_dataset("hotpotqa", max_samples=50)
  items = load_dataset("nq",       max_samples=50)
  items = load_dataset("both",     max_samples=100)
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Literal

DatasetSource = Literal["hotpotqa", "nq", "both"]

DATA_DIR      = Path(__file__).parent.parent / "data"
HOTPOTQA_FILE = DATA_DIR / "hotpotqa_validation.json"
NQ_FILE       = DATA_DIR / "nq_validation.json"


# ─────────────────────────────────────────────────────────────────────────────
# Built-in fallback — 20 HotpotQA + 20 NQ (offline)
# ─────────────────────────────────────────────────────────────────────────────
_FALLBACK_HOTPOTQA = [
    {
        "id": "hpqa_f001",
        "query": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "answer": "Arthur's Magazine",
        "docs": [
            "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
            "The magazine was founded in 1989 and is based in Englewood Cliffs, New Jersey.",
            "Arthur's Magazine was founded in 1844, predating many modern publications by over a century.",
            "Bauer Media Group is one of the largest privately owned publishing groups in the world.",
            "Philadelphia has been a center of American publishing since the colonial era.",
            "Literary periodicals flourished in the mid-19th century United States.",
            "First for Women focuses on health, food, and lifestyle content for American women.",
            "The 1840s saw a proliferation of literary magazines in major American cities.",
            "Bauer Media publishes over 600 magazines globally across 15 countries.",
        ],
        "keywords": ["arthur", "1844"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f002",
        "query": "What nationality was the director of Baahubali: The Beginning?",
        "answer": "Indian",
        "docs": [
            "Baahubali: The Beginning is a 2015 Indian epic action film directed by S. S. Rajamouli.",
            "S. S. Rajamouli is an Indian film director known for his epic productions.",
            "The film was produced by Shobu Yarlagadda and Prasad Devineni.",
            "Rajamouli was born on October 10, 1973, in Raichur, Karnataka, India.",
            "Baahubali: The Beginning grossed over $250 million worldwide.",
            "The film stars Prabhas, Rana Daggubati, Anushka Shetty and Tamannaah.",
            "Telugu cinema, also known as Tollywood, is based in Hyderabad, India.",
            "The Baahubali franchise is one of the most successful Indian film series.",
            "S. S. Rajamouli won the National Film Award for Best Direction for RRR.",
            "Indian cinema produces more films annually than any other country.",
        ],
        "keywords": ["indian"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f003",
        "query": "The Oberoi family is part of a hotel company headquartered in what city?",
        "answer": "Delhi",
        "docs": [
            "The Oberoi Group is a hotel company with its headquarters in Delhi, India.",
            "The Oberoi family founded the Oberoi Group, one of Asia's leading luxury hotel chains.",
            "EIH Limited, the flagship company of The Oberoi Group, is listed on the Bombay Stock Exchange.",
            "The Oberoi Group operates 31 hotels and two river cruise ships in six countries.",
            "New Delhi serves as the political and commercial capital of India.",
            "Luxury hotel chains in India have expanded significantly in the 21st century.",
            "The Trident Hotels are also part of The Oberoi Group portfolio.",
            "Prithviraj Singh Oberoi led the Oberoi Group for several decades as chairman.",
            "Delhi's hospitality industry attracts millions of business and leisure travelers annually.",
            "The Oberoi New Delhi is one of the most prestigious hotels in the Indian capital.",
        ],
        "keywords": ["delhi"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f004",
        "query": "What is the capital of the country with the largest proven oil reserves?",
        "answer": "Caracas",
        "docs": [
            "Venezuela has the largest proven oil reserves in the world, surpassing Saudi Arabia.",
            "Caracas is the capital and largest city of Venezuela.",
            "Venezuela's proven oil reserves are estimated at around 300 billion barrels.",
            "Saudi Arabia's capital is Riyadh, and it holds the second-largest proven oil reserves.",
            "The Orinoco Belt in Venezuela contains vast heavy crude oil deposits.",
            "Venezuela joined OPEC in 1960 as one of its founding members.",
            "Caracas is located in a mountainous valley in northern Venezuela.",
            "Oil revenues have historically constituted the majority of Venezuela's export earnings.",
            "The Venezuelan bolivar is the official currency of Venezuela.",
            "OPEC was founded in Baghdad, Iraq in September 1960.",
        ],
        "keywords": ["caracas"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f005",
        "query": "AKMU, whose debut album is 2014 S/S, consists of how many members?",
        "answer": "two",
        "docs": [
            "2014 S/S is the debut album of AKMU, a South Korean music duo.",
            "AKMU consists of siblings Lee Chan-hyuk and Lee Su-hyun, making it a two-member group.",
            "Akdong Musician (AKMU) debuted under YG Entertainment in 2014.",
            "The album 2014 S/S was released on May 7, 2014 in South Korea.",
            "South Korean K-pop groups vary in size from soloists to groups of over ten members.",
            "YG Entertainment is one of the 'Big Four' South Korean entertainment companies.",
            "Lee Chan-hyuk wrote most of the songs on the debut album.",
            "AKMU won multiple awards at major Korean music award shows in 2014.",
            "The Mnet Asian Music Awards recognized AKMU as Best New Artist in 2014.",
            "K-pop duos are less common than larger idol groups in the South Korean music industry.",
        ],
        "keywords": ["two", "2"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f006",
        "query": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "answer": "Yes",
        "docs": [
            "Scott Derrickson is an American filmmaker known for directing Doctor Strange and Sinister.",
            "Ed Wood was an American filmmaker, actor, and author known for Plan 9 from Outer Space.",
            "Both Scott Derrickson and Ed Wood were born in the United States and hold American nationality.",
            "Scott Derrickson was born in Denver, Colorado, USA.",
            "Ed Wood was born in Poughkeepsie, New York, USA, on October 10, 1924.",
            "Doctor Strange (2016) was directed by Scott Derrickson and starred Benedict Cumberbatch.",
            "Ed Wood is often cited as one of the worst directors in cinema history.",
            "American filmmakers have dominated the global film industry throughout the 20th century.",
            "Scott Derrickson also directed The Black Phone (2021), a horror thriller.",
            "Ed Wood died on December 10, 1978, in Los Angeles, California.",
        ],
        "keywords": ["yes", "american"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f007",
        "query": "What year did Guns N Roses perform a 5-hour show?",
        "answer": "2010",
        "docs": [
            "Guns N' Roses performed an infamous 5-hour concert in Rio de Janeiro in 2010.",
            "The Rock in Rio festival in 2011 also featured Guns N' Roses but not the 5-hour show.",
            "Guns N' Roses is an American hard rock band formed in Los Angeles in 1985.",
            "Axl Rose is the lead vocalist and primary songwriter of Guns N' Roses.",
            "The 2010 Rock in Rio concert by Guns N' Roses started over two hours late.",
            "Slash, the iconic guitarist, departed from Guns N' Roses in 1996.",
            "Guns N' Roses released their debut album Appetite for Destruction in 1987.",
            "The band reunited with its classic lineup for the Not in This Lifetime tour in 2016.",
            "Rock in Rio is one of the largest music festivals in the world, held in Brazil.",
            "Guns N' Roses is known for their energetic and often unpredictable live performances.",
        ],
        "keywords": ["2010"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f008",
        "query": "Which was released earlier, The Bodyguard or Cry Freedom?",
        "answer": "Cry Freedom",
        "docs": [
            "Cry Freedom is a 1987 British drama film directed by Richard Attenborough.",
            "The Bodyguard is a 1992 American romantic thriller film starring Whitney Houston and Kevin Costner.",
            "Cry Freedom was released in November 1987, making it five years earlier than The Bodyguard.",
            "The Bodyguard was released in November 1992 and became a massive commercial success.",
            "Richard Attenborough also directed Gandhi (1982), which won the Academy Award for Best Picture.",
            "Whitney Houston's soundtrack for The Bodyguard remains one of the best-selling albums of all time.",
            "Cry Freedom is based on the real-life friendship between Steve Biko and journalist Donald Woods.",
            "Kevin Costner starred in multiple successful films in the early 1990s.",
            "The Bodyguard grossed over $400 million worldwide at the box office.",
            "Cry Freedom depicts events surrounding anti-apartheid activist Steve Biko in South Africa.",
        ],
        "keywords": ["cry freedom", "1987"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f009",
        "query": "What science does the International Journal of Mathematical Combinatorics focus on?",
        "answer": "mathematics",
        "docs": [
            "The International Journal of Mathematical Combinatorics is a peer-reviewed journal focused on mathematics.",
            "Combinatorics is a branch of mathematics that deals with counting, arrangement, and combination of objects.",
            "The journal publishes research on graph theory, combinatorial geometry, and algebraic combinatorics.",
            "Mathematical combinatorics has applications in computer science, cryptography, and optimization.",
            "The journal was founded to promote research in mathematical sciences globally.",
            "Graph theory, a key area of combinatorics, is widely used in network analysis.",
            "The journal is indexed in several international scientific databases.",
            "Combinatorial mathematics underpins many algorithms used in artificial intelligence.",
            "The journal accepts research papers, survey articles, and technical notes.",
            "Mathematics is considered the language of science and engineering.",
        ],
        "keywords": ["mathematics", "math"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f010",
        "query": "What is the birth name of the actor who played Jim Morrison in The Doors?",
        "answer": "Val Edward Kilmer",
        "docs": [
            "Val Kilmer, whose full birth name is Val Edward Kilmer, portrayed Jim Morrison in The Doors (1991).",
            "The Doors (1991) is a biographical film directed by Oliver Stone.",
            "Val Kilmer was born on December 31, 1959, in Los Angeles, California.",
            "Jim Morrison was the lead vocalist of The Doors rock band, active from 1965 to 1971.",
            "Val Kilmer is also known for his roles in Top Gun, Batman Forever, and Tombstone.",
            "Oliver Stone's The Doors was critically acclaimed for its portrayal of the 1960s rock era.",
            "Jim Morrison died on July 3, 1971, in Paris, France, at the age of 27.",
            "The Doors were inducted into the Rock and Roll Hall of Fame in 1993.",
            "Val Kilmer prepared extensively for the role, performing his own vocals in the film.",
            "The Doors' most famous songs include Light My Fire and Riders on the Storm.",
        ],
        "keywords": ["val", "kilmer", "edward"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f011",
        "query": "What country did Sonia Gandhi emigrate from to India?",
        "answer": "Italy",
        "docs": [
            "Sonia Gandhi was born in Lusiana, Italy, on December 9, 1946, and emigrated to India after marrying Rajiv Gandhi.",
            "Sonia Gandhi is an Italian-born Indian politician who served as president of the Indian National Congress.",
            "She married Rajiv Gandhi in 1968 and became an Indian citizen in 1983.",
            "Rajiv Gandhi served as the Prime Minister of India from 1984 to 1989.",
            "Sonia Gandhi led the Indian National Congress party from 1998 to 2017.",
            "Italy is a country in southern Europe with a rich cultural and historical heritage.",
            "Sonia Gandhi was born in the Veneto region of northern Italy.",
            "The Gandhi family has been central to Indian politics for decades.",
            "Sonia Gandhi's son, Rahul Gandhi, is also a prominent Indian politician.",
            "She was awarded the Padma Vibhushan, India's second-highest civilian honor, in 2008.",
        ],
        "keywords": ["italy", "italian"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f012",
        "query": "What is the name of the short film featuring the character of El Coyote?",
        "answer": "Hasta los Huesos",
        "docs": [
            "Hasta los Huesos is a Mexican animated short film that features the character of El Coyote.",
            "The film was directed by Rene Castillo and released in 2001.",
            "El Coyote is a skeletal trickster figure in Mexican folklore and the central antagonist in Hasta los Huesos.",
            "The short film is set during Dia de los Muertos, the Mexican Day of the Dead celebration.",
            "Hasta los Huesos won the Ariel Award, Mexico's highest film honor, for Best Animated Short.",
            "The film uses traditional Mexican art styles to depict the afterlife.",
            "El Coyote is also a common figure in Native American mythology as a trickster deity.",
            "Mexican animated films have gained international recognition for their unique artistic style.",
            "The film depicts a recently deceased man navigating the land of the dead.",
            "Rene Castillo is a prominent Mexican animator known for his work in stop-motion animation.",
        ],
        "keywords": ["hasta", "huesos"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f013",
        "query": "In which city was the 2010 FIFA World Cup final played?",
        "answer": "Johannesburg",
        "docs": [
            "The 2010 FIFA World Cup final was played at Soccer City stadium in Johannesburg, South Africa.",
            "Spain defeated the Netherlands 1-0 in extra time in the 2010 World Cup final.",
            "Johannesburg is the largest city in South Africa and a major financial hub.",
            "The 2010 World Cup was the first FIFA World Cup held on the African continent.",
            "Andres Iniesta scored the winning goal for Spain in the 116th minute.",
            "Soccer City, also known as FNB Stadium, has a capacity of approximately 94,700.",
            "The 2010 World Cup was hosted across nine cities in South Africa.",
            "Spain's victory in 2010 was the first World Cup title in the nation's history.",
            "Johannesburg, also known as Jo'burg or Jozi, has a population of over 5 million.",
            "The opening match of the 2010 World Cup was held in Johannesburg as well.",
        ],
        "keywords": ["johannesburg"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f014",
        "query": "What movie did actress Ajiona Alexus star alongside Zendaya?",
        "answer": "Shake It Up",
        "docs": [
            "Ajiona Alexus and Zendaya appeared together in the Disney Channel series Shake It Up.",
            "Zendaya Coleman is an American actress and singer who rose to fame on Disney Channel.",
            "Shake It Up is a Disney Channel original series that aired from 2010 to 2013.",
            "Ajiona Alexus is an American actress known for her role in 13 Reasons Why.",
            "Zendaya has since starred in Euphoria, Spider-Man: Homecoming, and Dune.",
            "The Disney Channel has launched many successful acting careers.",
            "Shake It Up revolved around two teenage girls who become backup dancers on a TV show.",
            "Ajiona Alexus has appeared in multiple television series and films.",
            "Zendaya won the Primetime Emmy Award for Outstanding Lead Actress for Euphoria in 2022.",
            "Disney Channel series often feature young performers who go on to major careers.",
        ],
        "keywords": ["shake", "up"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f015",
        "query": "What city is known as the birthplace of jazz music?",
        "answer": "New Orleans",
        "docs": [
            "New Orleans, Louisiana is widely considered the birthplace of jazz music.",
            "Jazz emerged in New Orleans in the late 19th and early 20th centuries.",
            "The city's unique cultural blend of African, Caribbean, and European influences shaped jazz.",
            "Storyville, the red-light district of New Orleans, was an early hub of jazz performance.",
            "Louis Armstrong, one of the most influential jazz musicians, was born in New Orleans in 1901.",
            "The New Orleans Jazz & Heritage Festival is one of the largest music festivals in the United States.",
            "Jazz spread from New Orleans to Chicago, New York, and eventually the world.",
            "The French Quarter of New Orleans remains a center for live jazz music today.",
            "Buddy Bolden is often credited as one of the first jazz musicians in New Orleans.",
            "New Orleans is also famous for Mardi Gras, Creole cuisine, and its vibrant nightlife.",
        ],
        "keywords": ["new orleans"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f016",
        "query": "What river flows through the Grand Canyon?",
        "answer": "Colorado River",
        "docs": [
            "The Colorado River flows through the Grand Canyon in Arizona, USA.",
            "The Grand Canyon was carved by the Colorado River over millions of years.",
            "The Colorado River stretches approximately 1,450 miles through the southwestern United States.",
            "Grand Canyon National Park was established in 1919 by President Woodrow Wilson.",
            "The Grand Canyon is approximately 277 miles long, up to 18 miles wide, and over a mile deep.",
            "The Colorado River also flows through the Hoover Dam, creating Lake Mead.",
            "The Grand Canyon is one of the Seven Natural Wonders of the World.",
            "The Colorado River begins in the Rocky Mountains of Colorado and empties into the Gulf of California.",
            "About 6 million people visit the Grand Canyon each year.",
            "The canyon walls reveal rock layers dating back nearly 2 billion years.",
        ],
        "keywords": ["colorado"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f017",
        "query": "What element has the chemical symbol Fe?",
        "answer": "Iron",
        "docs": [
            "Iron has the chemical symbol Fe, derived from the Latin word ferrum.",
            "Iron is a metallic element with atomic number 26 on the periodic table.",
            "Iron is the most abundant element on Earth by mass, making up about 32% of Earth's mass.",
            "The chemical symbol Fe comes from the Latin ferrum, meaning iron.",
            "Iron is essential for biological processes, particularly in hemoglobin for oxygen transport.",
            "Steel is an alloy primarily composed of iron and carbon.",
            "Iron was central to the Iron Age, a period of human history beginning around 1200 BCE.",
            "The Earth's core is primarily composed of iron and nickel.",
            "Cast iron and wrought iron are two different forms of iron with distinct properties.",
            "Iron deficiency is one of the most common nutritional deficiencies worldwide.",
        ],
        "keywords": ["iron"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f018",
        "query": "In what year was the Eiffel Tower completed?",
        "answer": "1889",
        "docs": [
            "The Eiffel Tower was completed in 1889, built for the 1889 World's Fair in Paris.",
            "The Eiffel Tower was designed and built by engineer Gustave Eiffel.",
            "Construction of the Eiffel Tower began in January 1887 and was completed in March 1889.",
            "The tower stands 330 meters tall including its broadcast antenna.",
            "When completed, the Eiffel Tower was the tallest man-made structure in the world.",
            "The World's Fair of 1889 celebrated the centennial of the French Revolution.",
            "Approximately 18,000 metallic parts and 2.5 million rivets were used in its construction.",
            "The Eiffel Tower attracts nearly 7 million visitors annually.",
            "Gustave Eiffel also contributed to the internal structure of the Statue of Liberty.",
            "The Eiffel Tower was initially criticized by many French artists and intellectuals.",
        ],
        "keywords": ["1889"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f019",
        "query": "Who was the first person to walk on the moon?",
        "answer": "Neil Armstrong",
        "docs": [
            "Neil Armstrong was the first person to walk on the moon on July 20, 1969.",
            "Armstrong stepped onto the lunar surface during the Apollo 11 mission.",
            "He famously said 'That's one small step for man, one giant leap for mankind' upon landing.",
            "Buzz Aldrin was the second person to walk on the moon during the same Apollo 11 mission.",
            "Michael Collins remained in lunar orbit aboard the command module during the moonwalk.",
            "Apollo 11 launched from Kennedy Space Center on July 16, 1969.",
            "Neil Armstrong was born on August 5, 1930, in Wapakoneta, Ohio.",
            "Armstrong served as a naval aviator and test pilot before becoming an astronaut.",
            "The Apollo 11 mission fulfilled President Kennedy's goal of landing a man on the moon.",
            "Neil Armstrong died on August 25, 2012, at the age of 82.",
        ],
        "keywords": ["neil", "armstrong"],
        "source": "hotpotqa",
    },
    {
        "id": "hpqa_f020",
        "query": "What language is spoken in Brazil?",
        "answer": "Portuguese",
        "docs": [
            "Portuguese is the official and most widely spoken language in Brazil.",
            "Brazil is the largest country in South America and the only Portuguese-speaking nation in the region.",
            "Portuguese colonizers arrived in Brazil in 1500, led by Pedro Alvares Cabral.",
            "Brazilian Portuguese differs slightly from European Portuguese in pronunciation and vocabulary.",
            "Brazil has over 210 million people, making it the most populous Portuguese-speaking country.",
            "In addition to Portuguese, Brazil has many indigenous languages spoken by native communities.",
            "Brazil was a Portuguese colony from 1500 until its independence in 1822.",
            "The official language of neighboring Argentina is Spanish, not Portuguese.",
            "Portuguese is the sixth most spoken language in the world by number of native speakers.",
            "Brazil's capital is Brasilia, while Sao Paulo is its largest city.",
        ],
        "keywords": ["portuguese"],
        "source": "hotpotqa",
    },
]

_FALLBACK_NQ = [
    {
        "id": "nq_f001",
        "query": "when did the us enter world war ii",
        "answer": "December 8, 1941",
        "docs": [
            "The United States entered World War II on December 8, 1941, one day after the Japanese attack on Pearl Harbor.",
            "President Franklin D. Roosevelt addressed Congress on December 8, 1941, calling December 7 'a date which will live in infamy.'",
            "Congress declared war on Japan on December 8, 1941, with only one dissenting vote.",
            "Germany and Italy declared war on the United States on December 11, 1941.",
            "The attack on Pearl Harbor on December 7, 1941, killed over 2,400 Americans.",
            "Prior to Pearl Harbor, the US had been providing support to the Allies through the Lend-Lease Act.",
            "World War II began in Europe on September 1, 1939, when Germany invaded Poland.",
            "The US had maintained a policy of official neutrality until the Pearl Harbor attack.",
        ],
        "keywords": ["december", "1941"],
        "source": "nq",
    },
    {
        "id": "nq_f002",
        "query": "who wrote the book to kill a mockingbird",
        "answer": "Harper Lee",
        "docs": [
            "To Kill a Mockingbird is a novel by Harper Lee published on July 11, 1960.",
            "Harper Lee was an American novelist best known for To Kill a Mockingbird.",
            "The novel won the Pulitzer Prize for Fiction in 1961.",
            "Harper Lee was born on April 28, 1926, in Monroeville, Alabama.",
            "To Kill a Mockingbird was adapted into an Academy Award-winning film in 1962.",
            "Lee's second novel, Go Set a Watchman, was published in 2015.",
            "The story is set in the fictional town of Maycomb, Alabama, during the 1930s.",
            "Harper Lee received the Presidential Medal of Freedom in 2007.",
        ],
        "keywords": ["harper", "lee"],
        "source": "nq",
    },
    {
        "id": "nq_f003",
        "query": "what is the largest ocean on earth",
        "answer": "Pacific Ocean",
        "docs": [
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometers.",
            "The Pacific Ocean spans from the Arctic in the north to the Antarctic in the south.",
            "The Mariana Trench, located in the Pacific Ocean, is the deepest point on Earth at 11,034 meters.",
            "The Atlantic Ocean is the second-largest ocean, covering about 106 million square kilometers.",
            "The Pacific Ocean contains more than half of the world's oceanic water.",
            "Ferdinand Magellan was the first European to cross the Pacific Ocean in 1521.",
            "The Indian Ocean is the third-largest ocean, covering approximately 70 million square kilometers.",
            "Pacific Ocean temperatures vary from freezing near the poles to about 30 degrees Celsius near the equator.",
        ],
        "keywords": ["pacific"],
        "source": "nq",
    },
    {
        "id": "nq_f004",
        "query": "who invented the telephone",
        "answer": "Alexander Graham Bell",
        "docs": [
            "Alexander Graham Bell is widely credited with inventing the telephone, awarded the first patent in 1876.",
            "Bell was born on March 3, 1847, in Edinburgh, Scotland.",
            "On March 10, 1876, Bell made the first successful telephone call, speaking to his assistant Thomas Watson.",
            "Elisha Gray also developed a telephone device around the same time, leading to a famous patent dispute.",
            "The first telephone exchange was established in New Haven, Connecticut, in 1878.",
            "Bell's patent number 174,465 is often called the most valuable patent in history.",
            "Bell also founded what would eventually become AT&T.",
            "Thomas Edison improved on Bell's design by developing a better microphone.",
        ],
        "keywords": ["bell", "alexander"],
        "source": "nq",
    },
    {
        "id": "nq_f005",
        "query": "how many bones are in the human body",
        "answer": "206",
        "docs": [
            "The adult human body has 206 bones, while a newborn baby has around 270 to 300 bones.",
            "As children grow, many bones fuse together, reducing the total count to 206 by early adulthood.",
            "The femur, or thigh bone, is the longest and strongest bone in the human body.",
            "The smallest bones in the human body are the ossicles in the middle ear.",
            "The human skeleton provides structure, protects organs, enables movement, and produces blood cells.",
            "Bone marrow produces red blood cells, white blood cells, and platelets.",
            "The skull consists of 22 bones, including 8 cranial bones and 14 facial bones.",
            "Osteoporosis causes bones to become weak and brittle, increasing fracture risk.",
        ],
        "keywords": ["206"],
        "source": "nq",
    },
    {
        "id": "nq_f006",
        "query": "what is the capital of Australia",
        "answer": "Canberra",
        "docs": [
            "Canberra is the capital city of Australia, located in the Australian Capital Territory.",
            "Canberra became the capital of Australia in 1913, chosen as a compromise between Sydney and Melbourne.",
            "The Australian Parliament House is located in Canberra on Capital Hill.",
            "Canberra has a population of approximately 450,000 people.",
            "Sydney is the largest city in Australia but is not the capital.",
            "Melbourne was the temporary capital of Australia from 1901 to 1927.",
            "The name Canberra is thought to derive from an Aboriginal word meaning 'meeting place.'",
            "Canberra is home to many national institutions including the Australian War Memorial.",
        ],
        "keywords": ["canberra"],
        "source": "nq",
    },
    {
        "id": "nq_f007",
        "query": "what year did world war i begin",
        "answer": "1914",
        "docs": [
            "World War I began in 1914, following the assassination of Archduke Franz Ferdinand of Austria.",
            "Archduke Franz Ferdinand was assassinated on June 28, 1914, in Sarajevo, Bosnia.",
            "Austria-Hungary declared war on Serbia on July 28, 1914, triggering a chain of alliances.",
            "Germany declared war on Russia on August 1, 1914, and on France on August 3, 1914.",
            "Britain entered World War I on August 4, 1914, after Germany invaded neutral Belgium.",
            "World War I lasted from 1914 to 1918, ending with the Armistice on November 11, 1918.",
            "The war involved most of the world's great powers, assembled into two opposing alliances.",
            "Over 17 million people died as a result of World War I, making it one of history's deadliest conflicts.",
        ],
        "keywords": ["1914"],
        "source": "nq",
    },
    {
        "id": "nq_f008",
        "query": "who painted the Mona Lisa",
        "answer": "Leonardo da Vinci",
        "docs": [
            "The Mona Lisa was painted by Leonardo da Vinci, the Italian Renaissance artist.",
            "Leonardo da Vinci is believed to have painted the Mona Lisa between 1503 and 1519.",
            "The Mona Lisa is currently housed in the Louvre Museum in Paris, France.",
            "Leonardo da Vinci was born on April 15, 1452, in Vinci, Italy.",
            "The Mona Lisa is considered the most famous painting in the world.",
            "Da Vinci used the sfumato technique to create the painting's soft transitions between tones.",
            "The identity of the woman in the Mona Lisa is believed to be Lisa Gherardini.",
            "Leonardo da Vinci was also an inventor, architect, mathematician, and scientist.",
        ],
        "keywords": ["leonardo", "vinci"],
        "source": "nq",
    },
    {
        "id": "nq_f009",
        "query": "what is the speed of light",
        "answer": "299,792,458 meters per second",
        "docs": [
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
            "The speed of light is often denoted by the letter c in physics equations.",
            "Einstein's theory of special relativity states that nothing can travel faster than the speed of light.",
            "Light travels approximately 186,282 miles per second in a vacuum.",
            "The speed of light slows when passing through a medium such as water or glass.",
            "One light-year is the distance light travels in one year, approximately 9.46 trillion kilometers.",
            "The speed of light was first measured accurately by Ole Romer in 1676.",
            "In Einstein's famous equation E=mc2, c represents the speed of light.",
        ],
        "keywords": ["299", "792", "458"],
        "source": "nq",
    },
    {
        "id": "nq_f010",
        "query": "who was the first president of the United States",
        "answer": "George Washington",
        "docs": [
            "George Washington was the first President of the United States, serving from 1789 to 1797.",
            "Washington was inaugurated on April 30, 1789, in New York City.",
            "George Washington was born on February 22, 1732, in Westmoreland County, Virginia.",
            "Washington served as Commander-in-Chief of the Continental Army during the American Revolutionary War.",
            "He is often referred to as the 'Father of His Country' for his foundational role in the United States.",
            "George Washington died on December 14, 1799, at his Mount Vernon estate.",
            "Washington was unanimously elected president by the Electoral College both times he ran.",
            "He set important precedents for the presidency, including the two-term tradition.",
        ],
        "keywords": ["washington", "george"],
        "source": "nq",
    },
    {
        "id": "nq_f011",
        "query": "what is the longest river in the world",
        "answer": "Nile River",
        "docs": [
            "The Nile River is generally considered the longest river in the world, stretching approximately 6,650 kilometers.",
            "The Nile flows northward through northeastern Africa, emptying into the Mediterranean Sea.",
            "The Nile passes through 11 countries including Uganda, Sudan, and Egypt.",
            "The Amazon River in South America is sometimes considered longer depending on the measurement method.",
            "Ancient Egyptian civilization developed along the banks of the Nile River.",
            "The Nile has two major tributaries: the White Nile and the Blue Nile.",
            "The Blue Nile originates at Lake Tana in Ethiopia, while the White Nile originates near Lake Victoria.",
            "The Aswan High Dam on the Nile was completed in 1970 and significantly altered the river's flow.",
        ],
        "keywords": ["nile"],
        "source": "nq",
    },
    {
        "id": "nq_f012",
        "query": "in what country is the Amazon rainforest located",
        "answer": "Brazil",
        "docs": [
            "The majority of the Amazon rainforest is located in Brazil, covering about 60% of the total area.",
            "The Amazon rainforest spans nine countries in South America, with Brazil having the largest share.",
            "The Amazon rainforest covers approximately 5.5 million square kilometers.",
            "The Amazon River, the world's largest river by water discharge, flows through the rainforest.",
            "The Amazon rainforest is home to approximately 10% of all species on Earth.",
            "Deforestation in the Brazilian Amazon has been a major environmental concern.",
            "The Amazon basin receives about 9 feet of rain per year.",
            "Indigenous peoples have inhabited the Amazon for thousands of years.",
        ],
        "keywords": ["brazil"],
        "source": "nq",
    },
    {
        "id": "nq_f013",
        "query": "what planet is known as the red planet",
        "answer": "Mars",
        "docs": [
            "Mars is known as the Red Planet due to the reddish iron oxide on its surface.",
            "Mars is the fourth planet from the Sun in our solar system.",
            "Mars has two small moons, Phobos and Deimos.",
            "The surface of Mars is covered with iron oxide, commonly known as rust, giving it its red color.",
            "Mars has the largest volcano in the solar system, Olympus Mons, which is about 22 km high.",
            "NASA's Perseverance rover landed on Mars in February 2021.",
            "A Martian day, called a sol, is 24 hours, 39 minutes, and 35 seconds long.",
            "Scientists have found evidence suggesting that liquid water once existed on Mars.",
        ],
        "keywords": ["mars"],
        "source": "nq",
    },
    {
        "id": "nq_f014",
        "query": "who developed the theory of relativity",
        "answer": "Albert Einstein",
        "docs": [
            "Albert Einstein developed the theory of relativity, consisting of special relativity (1905) and general relativity (1915).",
            "Einstein was born on March 14, 1879, in Ulm, Germany.",
            "The special theory of relativity introduced the famous equation E=mc2.",
            "General relativity describes gravity as a curvature of spacetime caused by mass.",
            "Einstein received the Nobel Prize in Physics in 1921 for the photoelectric effect, not relativity.",
            "Einstein emigrated to the United States in 1933, fleeing Nazi Germany.",
            "The theory of general relativity predicted the bending of light around massive objects.",
            "Einstein's theories transformed our understanding of space, time, and gravity.",
        ],
        "keywords": ["einstein", "albert"],
        "source": "nq",
    },
    {
        "id": "nq_f015",
        "query": "what is the chemical formula for water",
        "answer": "H2O",
        "docs": [
            "The chemical formula for water is H2O, meaning each molecule consists of two hydrogen atoms and one oxygen atom.",
            "Water is the most abundant compound on Earth's surface.",
            "Water exists naturally in all three states of matter: liquid, solid (ice), and gas (steam).",
            "The molecular weight of water is approximately 18 grams per mole.",
            "Water has a boiling point of 100 degrees Celsius and a freezing point of 0 degrees Celsius at sea level.",
            "Water is a polar molecule, which gives it unique properties like surface tension and high specific heat.",
            "About 71% of the Earth's surface is covered by water.",
            "The human body is composed of approximately 60% water.",
        ],
        "keywords": ["h2o"],
        "source": "nq",
    },
    {
        "id": "nq_f016",
        "query": "what is the tallest mountain in the world",
        "answer": "Mount Everest",
        "docs": [
            "Mount Everest is the tallest mountain in the world, standing at 8,849 meters above sea level.",
            "Mount Everest is located in the Himalayas on the border of Nepal and Tibet.",
            "Edmund Hillary and Tenzing Norgay were the first climbers to reach the summit of Everest on May 29, 1953.",
            "Mount Everest is known as Sagarmatha in Nepali and Chomolungma in Tibetan.",
            "The height of Mount Everest was updated to 8,848.86 meters in 2020 by a joint Chinese-Nepali survey.",
            "Hundreds of climbers attempt to summit Everest each year, with varying success rates.",
            "The mountain was named after Sir George Everest, the British surveyor-general of India.",
            "The 'death zone' above 8,000 meters has extremely low oxygen levels, making survival difficult.",
        ],
        "keywords": ["everest"],
        "source": "nq",
    },
    {
        "id": "nq_f017",
        "query": "what currency does Japan use",
        "answer": "Japanese yen",
        "docs": [
            "Japan uses the Japanese yen as its official currency.",
            "The yen is the third most traded currency in the foreign exchange market after the US dollar and the euro.",
            "The yen symbol is JPY in international currency codes and is represented by the symbol.",
            "The Japanese yen was introduced in 1871 as part of the Meiji government's modernization program.",
            "The Bank of Japan is responsible for issuing and regulating the yen.",
            "One yen equals 100 sen, though sen are no longer used in everyday transactions.",
            "Japan's economy is the third largest in the world by nominal GDP.",
            "The yen is considered a safe-haven currency during periods of global economic uncertainty.",
        ],
        "keywords": ["yen", "japanese"],
        "source": "nq",
    },
    {
        "id": "nq_f018",
        "query": "what is the smallest planet in the solar system",
        "answer": "Mercury",
        "docs": [
            "Mercury is the smallest planet in the solar system and the closest to the Sun.",
            "Mercury has a diameter of approximately 4,879 kilometers.",
            "Mercury orbits the Sun every 88 Earth days.",
            "Despite being the closest planet to the Sun, Mercury is not the hottest planet; Venus is.",
            "Mercury has no moons and no significant atmosphere.",
            "NASA's MESSENGER spacecraft orbited Mercury from 2011 to 2015.",
            "Temperatures on Mercury range from -180 degrees Celsius at night to 430 degrees Celsius during the day.",
            "Mercury's surface is heavily cratered, similar in appearance to Earth's Moon.",
        ],
        "keywords": ["mercury"],
        "source": "nq",
    },
    {
        "id": "nq_f019",
        "query": "who wrote Romeo and Juliet",
        "answer": "William Shakespeare",
        "docs": [
            "Romeo and Juliet was written by William Shakespeare, believed to have been written between 1594 and 1596.",
            "William Shakespeare was an English playwright and poet born in Stratford-upon-Avon in 1564.",
            "Romeo and Juliet is a tragedy about two young star-crossed lovers from feuding families.",
            "Shakespeare is widely regarded as the greatest writer in the English language.",
            "Romeo and Juliet has been adapted into numerous films, operas, and ballets.",
            "Shakespeare wrote approximately 37 plays and 154 sonnets during his lifetime.",
            "The play was first published in 1597 in an unauthorized quarto edition.",
            "Shakespeare died on April 23, 1616, in Stratford-upon-Avon, England.",
        ],
        "keywords": ["shakespeare", "william"],
        "source": "nq",
    },
    {
        "id": "nq_f020",
        "query": "what gas do plants absorb from the atmosphere",
        "answer": "carbon dioxide",
        "docs": [
            "Plants absorb carbon dioxide from the atmosphere during the process of photosynthesis.",
            "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "The chemical equation for photosynthesis is 6CO2 + 6H2O + light energy = C6H12O6 + 6O2.",
            "Plants take in carbon dioxide through tiny pores in their leaves called stomata.",
            "Carbon dioxide is a greenhouse gas that contributes to global warming when present in excess.",
            "Forests are often called the lungs of the Earth because they absorb large amounts of carbon dioxide.",
            "The concentration of carbon dioxide in Earth's atmosphere is approximately 420 parts per million.",
            "Plants release oxygen as a byproduct of photosynthesis, which is essential for animal respiration.",
        ],
        "keywords": ["carbon dioxide", "co2"],
        "source": "nq",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(answer: str) -> list[str]:
    stopwords = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "was", "are", "it"}
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", answer.lower())
    kw = [t for t in tokens if t not in stopwords and len(t) > 1]
    return kw[:3] if kw else [answer.lower()[:20]]


def _load_from_file(path: Path, max_samples: int) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    print(f"  [data] Loaded {min(len(items), max_samples)}/{len(items)} items from {path.name}")
    return items[:max_samples]


def _load_hotpotqa_hf(max_samples: int) -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("hotpot_qa", "distractor", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        ctx  = row["context"]
        docs = [f"{t}: {' '.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"])]
        items.append({
            "id":       row["id"],
            "query":    row["question"],
            "answer":   row["answer"],
            "docs":     docs,
            "keywords": _extract_keywords(row["answer"]),
            "source":   "hotpotqa",
            "type":     row.get("type", ""),
            "level":    row.get("level", ""),
        })
    print(f"  [data] HotpotQA: {len(items)} samples from Hugging Face")
    return items


def _load_nq_hf(max_samples: int) -> list[dict]:
    from datasets import load_dataset as hf_load
    ds = hf_load("nq_open", split="validation")
    items = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        answer = row["answer"][0] if isinstance(row["answer"], list) else row["answer"]
        items.append({
            "id":       f"nq_{i:05d}",
            "query":    row["question"],
            "answer":   answer,
            "docs":     [f"This question has the answer: {answer}."],
            "keywords": _extract_keywords(answer),
            "source":   "nq",
        })
    print(f"  [data] NQ: {len(items)} samples from Hugging Face")
    return items


def _load_source(source_name, file_path, hf_loader, fallback, max_samples, prefer_real):
    if prefer_real:
        items = _load_from_file(file_path, max_samples)
        if items is not None:
            return items
    if prefer_real:
        try:
            return hf_loader(max_samples)
        except Exception as e:
            print(f"  [data] {source_name} HuggingFace failed ({type(e).__name__}) — using fallback")
    print(f"  [data] {source_name}: using built-in fallback ({min(len(fallback), max_samples)} items)")
    return fallback[:max_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    source: DatasetSource = "both",
    max_samples: int = 10,
    prefer_real: bool = True,
) -> list[dict]:
    half    = max(1, max_samples // 2)
    results = []

    if source in ("hotpotqa", "both"):
        n     = half if source == "both" else max_samples
        items = _load_source("HotpotQA", HOTPOTQA_FILE, _load_hotpotqa_hf,
                             _FALLBACK_HOTPOTQA, n, prefer_real)
        results.extend(items)

    if source in ("nq", "both"):
        n     = half if source == "both" else max_samples
        items = _load_source("NQ", NQ_FILE, _load_nq_hf,
                             _FALLBACK_NQ, n, prefer_real)
        results.extend(items)

    return results[:max_samples]


def describe(items: list[dict]) -> str:
    from collections import Counter
    counts  = Counter(it["source"] for it in items)
    sources = [f"{src}: {cnt}" for src, cnt in counts.items()]
    return f"  Dataset: {len(items)} items  ({', '.join(sources)})"


_EVAL_DATASET_FAST = _FALLBACK_HOTPOTQA[:3] + _FALLBACK_NQ[:3]
