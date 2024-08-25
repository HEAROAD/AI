import json
import os

# 메타데이터 파일이 있는 디렉토리
metadata_directory = '/Users/simmee/Documents/GitHub/AI/Data_03/metadata_03'
# 키포인트 폴더가 있는 디렉토리
keypoint_directory = '/Users/simmee/Documents/GitHub/AI/Data_03/keypoint_03'
# 결과를 저장할 디렉토리
output_directory = '/Users/simmee/Documents/GitHub/AI/output_03'

# 결과를 저장할 디렉토리가 없으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 메타데이터 파일 읽기
metadata_files = [f for f in os.listdir(metadata_directory) if f.endswith('.json')]

for metadata_file in metadata_files:
    metadata_file_path = os.path.join(metadata_directory, metadata_file)
    
    # 메타데이터 JSON 파일 읽기
    with open(metadata_file_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
        
        # 메타데이터에서 name을 가져옴
        if 'metaData' in metadata and 'name' in metadata['metaData']:
            video_name = metadata['metaData']['name']
            if 'data' in metadata and len(metadata['data']) > 0:
                attribute_name = metadata['data'][0]['attributes'][0]['name']
            else:
                print(f"No 'attributes' found in {metadata_file}")
                continue
            
            base_name = "_".join(video_name.split('_')[:3])  # 파일명의 첫 부분 추출 (예: NIA_SL_WORD0001)
            
            # 키포인트 폴더 찾기 (폴더 이름의 앞부분과 매칭)
            keypoint_folders = [f for f in os.listdir(keypoint_directory) if f.startswith(base_name)]
            print(f"Processing base_name: {base_name}, found folders: {keypoint_folders}")
            
            if keypoint_folders:
                keypoints = []
                
                for keypoint_folder in keypoint_folders:
                    keypoint_folder_path = os.path.join(keypoint_directory, keypoint_folder)
                    
                    if os.path.isdir(keypoint_folder_path):  # 폴더가 있는지 확인
                        # 키포인트 폴더 내의 JSON 파일들 탐색
                        keypoint_files = [f for f in os.listdir(keypoint_folder_path) if f.endswith('.json')]
                        
                        for keypoint_file in keypoint_files:
                            keypoint_file_path = os.path.join(keypoint_folder_path, keypoint_file)
                            print(f"Processing keypoint file: {keypoint_file}")
                            
                            # 키포인트 JSON 파일 읽기
                            with open(keypoint_file_path, 'r', encoding='utf-8') as kp_file:
                                keypoint_data = json.load(kp_file)
                                
                                # 'people'이 객체로 제공되는 경우 처리
                                if 'people' in keypoint_data and isinstance(keypoint_data['people'], dict):
                                    person = keypoint_data['people']
                                    hand_left_keypoints = person.get('hand_left_keypoints_2d', [])
                                    hand_right_keypoints = person.get('hand_right_keypoints_2d', [])
                                    
                                    # 데이터가 제대로 있는지 확인하기 위해 로그 출력
                                    print(f"File: {keypoint_file} processed successfully.")
                            
                                    keypoints.append({
                                        "file": keypoint_file,
                                        "hand_left_keypoints_2d": hand_left_keypoints,
                                        "hand_right_keypoints_2d": hand_right_keypoints
                                    })
                                else:
                                    print(f"No valid 'people' data found in keypoint file: {keypoint_file}")
                    else:
                        print(f"Keypoint folder is not a directory: {keypoint_folder_path}")
                
                if keypoints:
                    output_data = {
                        "word": attribute_name,  # 메타데이터의 'name' 값 사용
                        "keypoints": keypoints
                    }
                    
                    # JSON 파일로 저장
                    output_file_path = os.path.join(output_directory, f"{attribute_name}_keypoints.json")
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        json.dump(output_data, output_file, ensure_ascii=False, indent=4)
            else:
                print(f"Keypoint folder not found for base_name: {base_name}")
        else:
            print(f"{metadata_file} does not contain a valid 'metaData' or 'name'")
