# Spark-DSG (우리 사본) — 바꾼 것

원본: MIT-SPARK/Spark-DSG 3c40997, v1.1.3 (BSD-3). 원본 백업(변경 전): `Map_Vla/.build/spark_dsg_orig` (빌드 폴더라 지워질 수 있음).

## 끝난 것 (C++, 시험 133개 통과: `cmake -S . -B ../../.build/spark_dsg_base -DBUILD_TESTING=ON -DSPARK_DSG_BUILD_EXAMPLES=OFF`)
- 층: OBJECTS(+AGENTS partition), PLACES, ROOMS만. SEGMENTS, MESH_PLACES, TRAVERSABILITY, BUILDINGS 제거. 기본 그래프 층 키 {2, 3, 4}.
- 노드 속성: Place2d, Khronos, Traversability(+BoundaryInfo, traversability_boundary), NearestVertexInfo, ObjectNodeAttributes.mesh_connections 제거.
  PlaceNodeAttributes는 `distance`만(frontier, GVD basis points, mesh 연결, real_place 등 제거).
- mesh 전체 제거(mesh.h/.cpp, 직렬화, DynamicSceneGraph::setMesh/mesh, include_mesh 인자, BoundingBox의 Mesh 생성자). zmq 인터페이스 제거.
- 1.1.1 이전 형식(층 이름 표 없음) 읽기 제거. 줄인 형식으로 쓴 scene.json은 업스트림 파이썬 spark_dsg도 읽는다(확인함).

## 하다 만 것 (파이썬)
- `python/bindings`: attributes, scene_graph, spark_types, metadata, bindings 쪽은 새 C++에 맞게 고쳤고 mesh 바인딩은 지웠다. **빌드해 보지 않았다.**
- `python/spark_dsg`: zmq.py, networkx.py, torch_conversion.py, mp3d.py, commands/ 는 아직 안 지웠다. `__init__.py`가 torch_conversion을 import 하므로 지우면 같이 고쳐야 한다. viser.py 수정과 pyproject 의존성(click, shapely, zmq) 정리도 남았다.
