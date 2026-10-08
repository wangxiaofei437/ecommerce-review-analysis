/* ============================================================
 * Three.js 背景 - 银河粒子 + 旋转网格地球
 * 全屏 canvas#bg3d，鼠标视差跟随
 * ============================================================ */
(function () {
  const canvas = document.getElementById('bg3d');
  if (!canvas || typeof THREE === 'undefined') return;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x05091a, 0.0018);

  const camera = new THREE.PerspectiveCamera(
    60, window.innerWidth / window.innerHeight, 0.1, 5000
  );
  camera.position.set(0, 0, 360);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000, 0);

  // ---------- 1. 星空粒子（远） ----------
  const starGeo = new THREE.BufferGeometry();
  const starCount = 1500;
  const starPos = new Float32Array(starCount * 3);
  const starColor = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i++) {
    const r = 800 + Math.random() * 1200;
    const t = Math.random() * Math.PI * 2;
    const p = Math.acos(2 * Math.random() - 1);
    starPos[i * 3 + 0] = r * Math.sin(p) * Math.cos(t);
    starPos[i * 3 + 1] = r * Math.sin(p) * Math.sin(t);
    starPos[i * 3 + 2] = r * Math.cos(p);
    const c = 0.6 + Math.random() * 0.4;
    starColor[i * 3 + 0] = 0.5 * c;
    starColor[i * 3 + 1] = 0.8 * c;
    starColor[i * 3 + 2] = 1.0 * c;
  }
  starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
  starGeo.setAttribute('color',    new THREE.BufferAttribute(starColor, 3));
  const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
    size: 1.6, vertexColors: true, transparent: true,
    opacity: 0.9, depthWrite: false, sizeAttenuation: true,
  }));
  scene.add(stars);

  // ---------- 2. 中近层流动粒子云 ----------
  const cloudCount = 600;
  const cloudGeo = new THREE.BufferGeometry();
  const cloudPos = new Float32Array(cloudCount * 3);
  const cloudVel = new Float32Array(cloudCount * 3);
  for (let i = 0; i < cloudCount; i++) {
    cloudPos[i * 3 + 0] = (Math.random() - 0.5) * 800;
    cloudPos[i * 3 + 1] = (Math.random() - 0.5) * 500;
    cloudPos[i * 3 + 2] = (Math.random() - 0.5) * 600 - 100;
    cloudVel[i * 3 + 0] = (Math.random() - 0.5) * 0.05;
    cloudVel[i * 3 + 1] = (Math.random() - 0.5) * 0.05;
    cloudVel[i * 3 + 2] = (Math.random() - 0.5) * 0.05;
  }
  cloudGeo.setAttribute('position', new THREE.BufferAttribute(cloudPos, 3));
  const cloud = new THREE.Points(cloudGeo, new THREE.PointsMaterial({
    color: 0x66bbff, size: 2.4, transparent: true,
    opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  scene.add(cloud);

  // ---------- 3. 中央旋转线框球 ----------
  const sphereGeo = new THREE.IcosahedronGeometry(60, 2);
  const wire = new THREE.LineSegments(
    new THREE.WireframeGeometry(sphereGeo),
    new THREE.LineBasicMaterial({
      color: 0x4ea3ff, transparent: true, opacity: 0.18,
    })
  );
  wire.position.set(0, 0, -200);
  scene.add(wire);

  // 第二层稍大的环
  const torusGeo = new THREE.TorusGeometry(120, 0.4, 8, 100);
  const torus = new THREE.Mesh(
    torusGeo,
    new THREE.MeshBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0.35 })
  );
  torus.position.set(0, 0, -240);
  torus.rotation.x = Math.PI / 3;
  scene.add(torus);

  // ---------- 鼠标视差 ----------
  let mx = 0, my = 0;
  window.addEventListener('mousemove', (e) => {
    mx = (e.clientX / window.innerWidth - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // ---------- 渲染循环 ----------
  function tick(t) {
    t = t * 0.001;

    stars.rotation.y = t * 0.02;
    stars.rotation.x = t * 0.01;

    // 粒子云轻微飘动
    const pos = cloudGeo.attributes.position.array;
    for (let i = 0; i < cloudCount; i++) {
      pos[i * 3 + 0] += cloudVel[i * 3 + 0];
      pos[i * 3 + 1] += cloudVel[i * 3 + 1];
      pos[i * 3 + 2] += cloudVel[i * 3 + 2];
      // 越界回卷
      if (pos[i * 3 + 0] >  400) pos[i * 3 + 0] = -400;
      if (pos[i * 3 + 0] < -400) pos[i * 3 + 0] =  400;
      if (pos[i * 3 + 1] >  300) pos[i * 3 + 1] = -300;
      if (pos[i * 3 + 1] < -300) pos[i * 3 + 1] =  300;
    }
    cloudGeo.attributes.position.needsUpdate = true;

    wire.rotation.y = t * 0.18;
    wire.rotation.x = t * 0.10;
    torus.rotation.z = t * 0.25;

    // 摄像机视差
    camera.position.x += (mx * 30 - camera.position.x) * 0.04;
    camera.position.y += (-my * 20 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, -150);

    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
